# === IMPORTACIÓN DE LIBRERÍAS ===
import pandas as pd                      # Para manejo de datos tabulares (leer CSV, manipular columnas)
import re                                # Para usar expresiones regulares en la limpieza de texto
import string                            # Para eliminar puntuación del texto
import numpy as np                       # Para operaciones numéricas, matrices, etc.
import pickle                          # Para guardar y cargar objetos de Python (como el modelo entrenado)

# Scikit-learn: herramientas de preprocesamiento y separación de datos
from sklearn.model_selection import train_test_split  # Divide datos en entrenamiento y prueba
from sklearn.preprocessing import LabelEncoder        # Convierte etiquetas de texto a números

# Keras (TensorFlow): herramientas para texto y redes neuronales
from tensorflow.keras.preprocessing.text import Tokenizer            # type: ignore # Convierte texto a secuencias numéricas
from tensorflow.keras.preprocessing.sequence import pad_sequences    # type: ignore # Rellena secuencias a longitud fija
from tensorflow.keras.models import Sequential                      # type: ignore # Modelo secuencial (de capas)
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout  # type: ignore # Capas del modelo
from tensorflow.keras.utils import to_categorical                   # type: ignore # Convierte etiquetas numéricas a one-hot

# === LIMPIEZA DE TEXTO ===

# Lista de palabras vacías (stop words) en español que eliminamos por no aportar valor semántico
stop_words = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo',
    'como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre',
    'también','me','hasta','hay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra',
    'otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra',
    'él','tanto','esa','estos','mucho','quienes','nada','muchos'
}

# Función que limpia y preprocesa el texto
def limpiar_texto(texto):
    texto = texto.lower()  # Convierte todo a minúsculas
    texto = re.sub(r'[^ -]+', '', texto)  # Elimina caracteres no ASCII
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # Elimina signos de puntuación
    palabras = texto.split()  # Divide el texto en palabras individuales
    # Elimina stopwords y palabras de menos de 3 letras
    return ' '.join([p for p in palabras if p not in stop_words and len(p) > 2])

# Clasifica el sentimiento según la calificación del comentario
def clasificar_sentimiento(row):
    if row['calificacion'] >= 4:
        return 'positivo'
    elif row['calificacion'] == 3:
        return 'neutro'
    else:
        return 'negativo'
    
# === CARGA Y PROCESAMIENTO DE DATOS ===

# Carga los datos desde un archivo CSV
df = pd.read_csv("datos/Dataset_Comentarios_con_Calificacion.csv")

# Limpia todos los comentarios del dataset
df['comentario_limpio'] = df['comentario'].apply(lambda x: limpiar_texto(str(x)))

# Clasifica cada comentario en positivo, neutro o negativo
df['sentimiento'] = df.apply(clasificar_sentimiento, axis=1)
print(df["sentimiento"].value_counts()) 
# === TOKENIZACIÓN DEL TEXTO ===

# Separa entradas (comentario limpio) y salidas (etiquetas)
X_text = df['comentario_limpio'].values
label_encoder = LabelEncoder()  # Codificador de etiquetas
y = label_encoder.fit_transform(df['sentimiento'])  # Convierte etiquetas a números

# Divide en datos de entrenamiento y prueba (80%-20%)
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Crea tokenizador para convertir texto a secuencias numéricas
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)  # Aprende el vocabulario

# Convierte los textos a secuencias de números
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Rellena o recorta secuencias a una longitud fija (100)
X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')

# Convierte etiquetas numéricas a codificación one-hot para clasificación múltiple
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# === MODELO DE RED NEURONAL ===

# Define el modelo neuronal secuencial
modelo = Sequential([
    Embedding(10000, 64, input_length=100),      # Capa de embedding para representar palabras como vectores
    GlobalAveragePooling1D(),                    # Promedia todos los vectores del comentario
    Dense(64, activation='relu'),                # Capa oculta con activación ReLU
    Dropout(0.3),                                # Capa para evitar overfitting (30% de apagado aleatorio)
    Dense(3, activation='softmax')               # Capa de salida con 3 neuronas (una por clase)
])

# Compila el modelo con función de pérdida y optimizador
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrena el modelo con los datos
modelo.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat), verbose=1)

# === DETECCIÓN DE PALABRAS OFENSIVAS ===

# Carga archivo CSV que contiene palabras ofensivas
ofensivas_df = pd.read_csv("datos/palabras_ofensivas.csv")

# Crea un conjunto con todas las palabras ofensivas en minúsculas
palabras_ofensivas = set(ofensivas_df["palabra"].dropna().str.lower())

# Función para detectar si un texto contiene alguna palabra ofensiva
def contiene_ofensas(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)  # Elimina signos de puntuación
    palabras = texto.split()
    return any(p in palabras_ofensivas for p in palabras)

# === PRUEBA EN VIVO ===

# Modo interactivo: el usuario escribe un comentario y se analiza
print("\n✅ Modelo entrenado. Escribe un comentario para analizar (o 'salir'):")

while True:
    entrada = input("Tu comentario: ")  # Captura comentario del usuario
    if entrada.lower() == 'salir':
        break  # Sale del bucle si el usuario escribe 'salir'

    limpio = limpiar_texto(entrada)  # Limpia el comentario
    secuencia = tokenizer.texts_to_sequences([limpio])  # Lo convierte en secuencia
    secuencia_pad = pad_sequences(secuencia, maxlen=100, padding='post')  # Lo rellena a longitud 100
    pred = modelo.predict(secuencia_pad, verbose=0)  # Hace la predicción
    clase = label_encoder.inverse_transform([np.argmax(pred)])  # Convierte de número a clase
    ofensivo = contiene_ofensas(entrada)  # Detecta si contiene lenguaje ofensivo

    # Muestra los resultados
    print(f"➡️  Sentimiento: {clase[0].upper()}")
    print(f"🚫 Ofensivo: {'Sí' if ofensivo else 'No'}\n")

# Fin del script
# Guarda el modelo, el tokenizador y el codificador
modelo.save("modelos/modelo.keras", save_format="keras")  # guarda en la carpeta correcta

with open("modelos/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("modelos/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
