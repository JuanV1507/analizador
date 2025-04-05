import pandas as pd
from transformers import pipeline
import os

# Ruta del CSV donde tienes los comentarios
csv_path = "datos/Dataset_Comentarios_con_Calificacion.csv"

# Verifica si el archivo existe
if not os.path.exists(csv_path):
    print("❌ No se encontró el archivo dataset_manual.csv en la carpeta 'datos/'.")
    exit()

# Cargar los comentarios
df = pd.read_csv(csv_path)

# Verifica si hay comentarios
if df.empty:
    print("⚠️ No hay comentarios en el archivo.")
    exit()

# Obtener los últimos 15 comentarios
comentarios = df['comentario'].dropna().tolist()
comentarios_recientes = comentarios[-15:] if len(comentarios) >= 15 else comentarios

# Unir los comentarios en una sola cadena
texto = " ".join(comentarios_recientes)

# Cargar modelo T5 para resumir
print("🧠 Generando reseña con T5...")
resumidor = pipeline("summarization", model="t5-base", tokenizer="t5-base")

# Generar la reseña
resumen = resumidor(texto, max_length=100, min_length=20, do_sample=False)[0]['summary_text']

# Mostrar la reseña generada
print("\n📝 Reseña generada automáticamente:")
print(resumen)
