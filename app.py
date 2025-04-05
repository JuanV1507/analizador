from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mysql.connector
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model # type: ignore
from datetime import datetime
import csv
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


app = Flask(__name__)
CORS(app)
#bloquear los insultos de la app--------------------


# Cargar palabras ofensivas desde CSV
def cargar_palabras_ofensivas():
    palabras = set()
    with open('datos/palabras_ofensivas.csv', newline='', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile):
            palabra = row[0].strip().lower()
            if palabra:
                palabras.add(palabra)
    return palabras

palabras_ofensivas = cargar_palabras_ofensivas()

# -------------------- Conexión MySQL --------------------
db = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="",
    database="bd_comentarios"
)
cursor = db.cursor()

# -------------------- Cargar modelo y tokenizer --------------------
modelo = load_model('modelos/modelo.h5')
tokenizer = pickle.load(open('modelos/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('modelos/label_encoder.pkl', 'rb'))

# -------------------- Función para limpiar texto --------------------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Záéíóúüñ0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# -------------------- Función para predecir sentimiento --------------------
def predecir_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    secuencia_padded = pad_sequences(secuencia, maxlen=100)  # 👈 IMPORTANTE
    pred = modelo.predict(secuencia_padded, verbose=0)
    etiqueta = label_encoder.inverse_transform([np.argmax(pred)])
    return etiqueta[0]

# -------------------- Función para generar reseña general --------------------
def generar_resena_general():
    cursor.execute("SELECT comentario FROM comentarios")
    comentarios = [r[0] for r in cursor.fetchall()]
    if not comentarios:
        return "Aún no hay suficientes comentarios para generar una reseña."
    texto_completo = " ".join(comentarios)
    return f"Reseña basada en {len(comentarios)} comentarios: \"{texto_completo[:300]}...\""

# -------------------- Ruta principal del dashboard --------------------
@app.route('/')
def dashboard():
    cursor.execute("SELECT * FROM comentarios ORDER BY id DESC")
    comentarios = cursor.fetchall()

    comentarios_list = []
    stats = {"positivo": 0, "neutro": 0, "negativo": 0}

    for c in comentarios:
        comentario_dict = {
            "id": c[0],
            "producto_id": c[1],
            "nombre_usuario": c[2],
            "comentario": c[3],
            "sentimiento": c[4],
            "fecha": c[5]
        }
        comentarios_list.append(comentario_dict)
        if c[4] in stats:
            stats[c[4]] += 1

    resumen = generar_resena_general()

    return render_template("dashboard.html", comentarios=comentarios_list, stats=stats, resumen=resumen)

# -------------------- Ruta para insertar un comentario --------------------
@app.route('/insertar_comentario', methods=['POST'])
def insertar_comentario():
    data = request.json if request.is_json else request.form

    producto_id = data.get("producto_id")
    nombre_usuario = data.get("nombre_usuario")
    comentario = data.get("comentario")
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Validación de campos
    if not producto_id or not nombre_usuario or not comentario:
        return jsonify({"error": "Faltan campos requeridos"}), 400

    # Verificar si el producto_id existe
    cursor.execute("SELECT COUNT(*) FROM productos WHERE id = %s", (producto_id,))
    if cursor.fetchone()[0] == 0:
        return jsonify({"error": "El producto_id no existe en la base de datos"}), 400

    # Predecir sentimiento
    # Verificar si es ofensivo
    if es_comentario_ofensivo(comentario):
     return jsonify({"error": "Comentario ofensivo bloqueado"}), 400

# Si no es ofensivo, predecir sentimiento
    sentimiento = predecir_sentimiento(comentario)

    # Insertar en base de datos
    try:
        cursor.execute(
            "INSERT INTO comentarios (producto_id, nombre_usuario, comentario, sentimiento, fecha) VALUES (%s, %s, %s, %s, %s)",
            (producto_id, nombre_usuario, comentario, sentimiento, fecha)
        )
        db.commit()
        return jsonify({"mensaje": "Comentario insertado exitosamente", "sentimiento": sentimiento}), 200
    except mysql.connector.Error as err:
        db.rollback()
        return jsonify({"error": f"Error al insertar comentario: {err}"}), 500

# -------------------- Ruta para obtener todos los comentarios --------------------
@app.route('/comentarios', methods=['GET'])
def obtener_comentarios():
    cursor.execute("SELECT * FROM comentarios ORDER BY id DESC")
    comentarios = cursor.fetchall()
    comentarios_list = []
    for c in comentarios:
        comentarios_list.append({
            "id": c[0],
            "producto_id": c[1],
            "nombre_usuario": c[2],
            "comentario": c[3],
            "sentimiento": c[4],
            "fecha": c[5]
        })
    return jsonify(comentarios_list)
#----funcion para los insultos--------
def es_comentario_ofensivo(texto):
    palabras = limpiar_texto(texto).split()
    return any(palabra in palabras_ofensivas for palabra in palabras)


# -------------------- Ejecutar app --------------------
if __name__ == '__main__':
    app.run(debug=True)
