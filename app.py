from flask import Flask, jsonify, redirect, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
import string
from transformers import pipeline
import csv
import os
import mysql.connector
from flask_cors import CORS

# === FLASK ===
app = Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return render_template("entrenar_manual.html")

@app.route("/guardar", methods=["GET", "POST"])
def guardar():
    if request.method == "POST":
        comentario = request.form["comentario"]
        # guardar lógica aquí
        return "Comentario guardado"
    else:
        return redirect("/")

# === CONEXIÓN A BASE DE DATOS ===
conexion = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="",
    database="bd_comentarios"
)


with open("modelos/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("modelos/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

ofensivas_df = pd.read_csv("datos/palabras_ofensivas.csv")
palabras_ofensivas = set(ofensivas_df["palabra"].dropna().str.lower())

# === STOPWORDS Y FUNCIONES ===
stop_words = {...}  # <- Puedes pegar aquí el mismo set que ya tienes

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

# === API PARA COMENTARIOS DESDE FRONT ===
@app.route('/api/comentarios', methods=['POST'])
def recibir_comentario():
    data = request.get_json()
    producto_id = data.get('producto_id')
    nombre_usuario = data.get('nombre_usuario')
    comentario = data.get('comentario', '')

    # Procesamiento del comentario
    limpio = limpiar_texto(comentario)
    secuencia = tokenizer.texts_to_sequences([limpio])
    secuencia_pad = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=100, padding='post')
    pred = modelo.predict(secuencia_pad, verbose=0)
    clase = label_encoder.inverse_transform([np.argmax(pred)])[0]
    ofensivo = contiene_ofensas(comentario)

    # Guardar en la base de datos
    cursor = conexion.cursor()
    cursor.execute(
        "INSERT INTO comentarios (producto_id, nombre_usuario, comentario, sentimiento) VALUES (%s, %s, %s, %s)",
        (producto_id, nombre_usuario, comentario, clase)
    )
    conexion.commit()
    cursor.close()

    return jsonify({
        "mensaje": "Comentario recibido y guardado",
        "sentimiento": clase,
        "ofensivo": ofensivo
    }), 201

# === ENTRENAMIENTO MANUAL CON RESEÑA AUTOMÁTICA ===
resumidor = pipeline("summarization", model="t5-small", tokenizer="t5-small")

@app.route("/entrenar-manual", methods=["GET", "POST"])
def entrenar_manual():
    mensaje = ""
    resumen_generado = ""
    archivo = "datos/dataset_manual.csv"
    encabezado = ["comentario", "sentimiento"]
    comentarios_existentes = []

    if os.path.isfile(archivo):
        with open(archivo, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            comentarios_existentes = [row["comentario"] for row in reader]

    if request.method == "POST":
        comentario = request.form["comentario"]
        sentimiento = request.form["sentimiento"]
        if comentario and sentimiento:
            existe = os.path.isfile(archivo)
            with open(archivo, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=encabezado)
                if not existe:
                    writer.writeheader()
                writer.writerow({"comentario": comentario, "sentimiento": sentimiento})
            mensaje = "✅ Comentario guardado exitosamente"
            comentarios_existentes.append(comentario)

    if comentarios_existentes:
        texto = " ".join(comentarios_existentes[-15:])
        resumen = resumidor(texto, max_length=100, min_length=20, do_sample=False)
        resumen_generado = resumen[0]['summary_text']

    return render_template("entrenar_manual.html", mensaje=mensaje, resumen=resumen_generado)


# === INICIO ===
if __name__ == '__main__':
    app.run(debug=True)

