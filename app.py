from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
import string
from transformers import pipeline
import csv
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# === CARGA MODELO Y UTILIDADES ===
modelo = tf.keras.models.load_model("modelo.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

ofensivas_df = pd.read_csv("palabras_ofensivas.csv")
palabras_ofensivas = set(ofensivas_df["palabra"].dropna().str.lower())

# === APP FLASK ===
app = Flask(__name__)

stop_words = { ... }  # Pega aquí el mismo set de stopwords de antes

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^ -]+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    return ' '.join([p for p in palabras if p not in stop_words and len(p) > 2])

def contiene_ofensas(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palabras = texto.split()
    return any(p in palabras_ofensivas for p in palabras)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    ofensivo = None
    if request.method == 'POST':
        comentario = request.form['comentario']
        limpio = limpiar_texto(comentario)
        secuencia = tokenizer.texts_to_sequences([limpio])
        secuencia_pad = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=100, padding='post')
        pred = modelo.predict(secuencia_pad, verbose=0)
        clase = label_encoder.inverse_transform([np.argmax(pred)])[0]
        ofensivo = contiene_ofensas(comentario)
        resultado = clase.upper()
    return render_template("index.html", resultado=resultado, ofensivo=ofensivo)

# Cargar modelo T5 una sola vez
resumidor = pipeline("summarization", model="t5-small", tokenizer="t5-small")

@app.route("/entrenar-manual", methods=["GET", "POST"])
def entrenar_manual():
    mensaje = ""
    resumen_generado = ""

    archivo = "datos/dataset_manual.csv"
    encabezado = ["comentario", "sentimiento"]

    # Cargar todos los comentarios existentes
    comentarios_existentes = []
    if os.path.isfile(archivo):
        with open(archivo, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            comentarios_existentes = [row["comentario"] for row in reader]

    # Si se está enviando un nuevo comentario
    if request.method == "POST":
        comentario = request.form["comentario"]
        sentimiento = request.form["sentimiento"]

        if comentario and sentimiento:
            # Guardar en CSV
            existe = os.path.isfile(archivo)
            with open(archivo, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=encabezado)
                if not existe:
                    writer.writeheader()
                writer.writerow({"comentario": comentario, "sentimiento": sentimiento})
            mensaje = "✅ Comentario guardado exitosamente"

            # Agregar el nuevo comentario a la lista
            comentarios_existentes.append(comentario)

    # Generar resumen automático
    if comentarios_existentes:
        texto = " ".join(comentarios_existentes[-15:])  # Tomar los últimos 15 para evitar exceso
        resumen = resumidor(texto, max_length=100, min_length=20, do_sample=False)
        resumen_generado = resumen[0]['summary_text']

    return render_template("entrenar_manual.html", mensaje=mensaje, resumen=resumen_generado)

if __name__ == '__main__':
    app.run(debug=True)
