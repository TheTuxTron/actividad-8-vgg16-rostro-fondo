"""
App de Streamlit para presentar la Actividad 8.

Ejecutar:
    streamlit run app.py
"""
import numpy as np
import cv2
import streamlit as st
from keras.models import load_model

from utils import a_tensor, cargar_clases, preparar_imagen

st.set_page_config(page_title="Rostro vs Fondo", page_icon="🧑", layout="centered")


@st.cache_resource
def cargar():
    return load_model("modelo_rostro_fondo.keras"), cargar_clases()


try:
    model, CLASES = cargar()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo. Entrena primero con el notebook.\n\n{e}")
    st.stop()

ESTUDIANTE = next(c for c in CLASES if c != "Fondo")

st.title("Clasificador VGG-16: Estudiante vs Fondo")
st.caption(f"Clases: **{ESTUDIANTE}** y **Fondo** · Transfer learning con VGG-16")

with st.sidebar:
    st.header("Ajustes")
    umbral = st.slider(
        f"Confianza mínima para decir «{ESTUDIANTE}»", 0.50, 0.99, 0.70, 0.01,
        help="Si la probabilidad del estudiante no supera este valor, se responde «Fondo». "
             "Subirlo ayuda en la prueba 3 (celebridad parecida).",
    )
    st.markdown(
        "**Pruebas**\n"
        f"1. Foto tuya con webcam → **{ESTUDIANTE}**\n"
        "2. Fondo de Google Images → **Fondo**\n"
        "3. Celebridad parecida → **Fondo**"
    )

tab_cam, tab_archivo = st.tabs(["📷 Webcam", "🖼️ Subir imagen"])
with tab_cam:
    foto = st.camera_input("Tómate una foto")
with tab_archivo:
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])

fuente = foto or archivo
if fuente is None:
    st.info("Toma una foto o sube una imagen para clasificarla.")
    st.stop()

img_bgr = cv2.imdecode(np.frombuffer(fuente.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("No se pudo leer la imagen.")
    st.stop()

# Guardamos cada imagen probada (útil para revisar fallos o reentrenar)
import os, time  # noqa: E401
os.makedirs("capturas_app", exist_ok=True)
cv2.imwrite(os.path.join("capturas_app", f"{'cam' if foto else 'archivo'}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"), img_bgr)

img, hay_rostro = preparar_imagen(img_bgr)
pred = model.predict(a_tensor(img), verbose=0)[0]
p_est = float(pred[CLASES.index(ESTUDIANTE)])
etiqueta = ESTUDIANTE if p_est >= umbral else "Fondo"

col1, col2 = st.columns(2)
with col1:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)
with col2:
    st.image(img, caption="Lo que ve el modelo" + (" (rostro recortado)" if hay_rostro else " (sin rostro detectado)"),
             use_container_width=True)

if etiqueta == ESTUDIANTE:
    st.success(f"## Predicción: {etiqueta}")
else:
    st.warning(f"## Predicción: {etiqueta}")

for c, p in zip(CLASES, pred):
    st.write(f"**{c}**: {p:.1%}")
    st.progress(float(p))
