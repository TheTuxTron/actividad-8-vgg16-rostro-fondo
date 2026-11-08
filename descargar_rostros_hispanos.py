"""
Toma del dataset LFW (ya descargado por descargar_rostros.py) rostros de personas
con nombres hispanos -> mayoría latinos/españoles, más parecidos al estudiante.
Sirve para la prueba 3 (celebridad parecida -> "Fondo").

Uso:
    python descargar_rostros_hispanos.py        # hasta 150 imágenes
"""
import os
import sys

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from utils import guardar_imagen

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
POR_PERSONA = 2

NOMBRES = {
    "Jose", "Juan", "Carlos", "Luis", "Jorge", "Miguel", "Pedro", "Alejandro",
    "Fernando", "Ricardo", "Eduardo", "Francisco", "Javier", "Antonio", "Manuel",
    "Rafael", "Raul", "Roberto", "Andres", "Diego", "Gustavo", "Hugo", "Oscar",
    "Pablo", "Sergio", "Mario", "Alvaro", "Hector", "Ernesto", "Enrique", "Felipe",
    "Gonzalo", "Ignacio", "Julio", "Lucio", "Marcelo", "Nestor", "Ramon", "Rodrigo",
    "Vicente", "Alberto", "Armando", "Cesar", "Daniel", "Gabriel", "Guillermo",
    "Gerardo", "Jaime", "Joaquin", "Martin", "Mauricio", "Nicolas", "Orlando",
    "Salvador", "Santiago", "Tomas", "Victor", "Alfonso", "Arturo", "Rafael",
    "Fidel", "Hugo", "Alan", "Alvaro", "Abel", "Lula", "Vladimiro", "Ruben",
}

destino = os.path.join("dataset", "Fondo")
os.makedirs(destino, exist_ok=True)

lfw = fetch_lfw_people(color=True, resize=1.0, slice_=None, min_faces_per_person=0)
imgs = lfw.images
if imgs.max() <= 1.0:
    imgs = imgs * 255.0
imgs = imgs.astype(np.uint8)

nombres_persona = lfw.target_names
elegidos = []
for pid, nombre in enumerate(nombres_persona):
    if nombre.split()[0] in NOMBRES:
        idx = np.where(lfw.target == pid)[0][:POR_PERSONA]
        elegidos.extend(idx.tolist())

rng = np.random.default_rng(7)
rng.shuffle(elegidos)
elegidos = elegidos[:N]

for i, idx in enumerate(elegidos, 1):
    img_bgr = cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR)
    guardar_imagen(os.path.join(destino, f"lfwhisp_{i:04d}.jpg"), img_bgr)

print(f"Guardados {len(elegidos)} rostros hispanos en {destino}")
