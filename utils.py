"""
Funciones compartidas entre el notebook de entrenamiento y la app de Streamlit.

IMPORTANTE: el modelo debe recibir las imágenes procesadas EXACTAMENTE igual
en el entrenamiento y en la predicción. Por eso todo el preprocesamiento vive aquí.
"""
import json
import os

import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input

IMG_SIZE = 224
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Si es True se recorta el rostro (si se detecta) antes de clasificar.
# Esto obliga al modelo a fijarse en la CARA y no en el fondo de la foto
# (p. ej. la pared de tu cuarto), lo cual es clave para la prueba 3
# (celebridad parecida a ti -> "Fondo").
USAR_RECORTE_ROSTRO = True

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detectar_rostro(img_bgr):
    """Devuelve (x, y, w, h) del rostro más grande o None si no hay rostro."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lado_min = max(30, min(gris.shape[:2]) // 10)
    rostros = _CASCADE.detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5, minSize=(lado_min, lado_min)
    )
    if len(rostros) == 0:
        return None
    return max(rostros, key=lambda r: r[2] * r[3])


def preparar_imagen(img_bgr, recortar=USAR_RECORTE_ROSTRO):
    """
    Recibe una imagen BGR (formato OpenCV) y devuelve:
      - imagen RGB uint8 de 224x224 lista para el modelo / para mostrar
      - bool indicando si se detectó (y recortó) un rostro
    """
    hay_rostro = False
    if recortar:
        caja = detectar_rostro(img_bgr)
        if caja is not None:
            x, y, w, h = caja
            margen = int(0.25 * max(w, h))
            H, W = img_bgr.shape[:2]
            x0, y0 = max(0, x - margen), max(0, y - margen)
            x1, y1 = min(W, x + w + margen), min(H, y + h + margen)
            img_bgr = img_bgr[y0:y1, x0:x1]
            hay_rostro = True
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, hay_rostro


def a_tensor(imgs_rgb):
    """Lista/array de imágenes RGB uint8 -> tensor normalizado para VGG-16."""
    x = np.asarray(imgs_rgb, dtype="float32")
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return preprocess_input(x)  # normalización propia de VGG-16 (ImageNet)


def leer_imagen(ruta):
    """cv2.imread no soporta rutas con tildes/ñ en Windows; esto sí."""
    datos = np.fromfile(ruta, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar_imagen(ruta, img_bgr):
    ext = os.path.splitext(ruta)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img_bgr)
    if ok:
        buf.tofile(ruta)
    return ok


def guardar_clases(clases, ruta="clases.json"):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(clases, f, ensure_ascii=False, indent=2)


def cargar_clases(ruta="clases.json"):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
