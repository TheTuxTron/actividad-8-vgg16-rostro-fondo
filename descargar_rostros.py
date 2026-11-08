"""
Descarga rostros de OTRAS personas (dataset público LFW - Labeled Faces in the Wild,
casi todos famosos) y los guarda en dataset/Fondo.

¿Por qué? La prueba 3 pide que la foto de una celebridad parecida a ti se
clasifique como "Fondo". Si el modelo solo vio paredes y paisajes como "Fondo",
aprendería "cara = estudiante". Con caras de otras personas en "Fondo" aprende
a distinguir TU cara de las demás.

Uso:
    python descargar_rostros.py          # 300 rostros (por defecto)
    python descargar_rostros.py 500
La primera vez descarga ~200 MB.
"""
import os
import sys

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from utils import guardar_imagen

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
destino = os.path.join("dataset", "Fondo")
os.makedirs(destino, exist_ok=True)

print("Descargando LFW (puede tardar la primera vez)...")
lfw = fetch_lfw_people(color=True, resize=1.0, slice_=None, min_faces_per_person=0)
imgs = lfw.images
if imgs.max() <= 1.0:
    imgs = imgs * 255.0
imgs = imgs.astype(np.uint8)

# Una foto por persona para maximizar la variedad de caras
rng = np.random.default_rng(42)
_, idx_por_persona = np.unique(lfw.target, return_index=True)
rng.shuffle(idx_por_persona)
seleccion = idx_por_persona[:N]

for i, idx in enumerate(seleccion, 1):
    img_bgr = cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR)
    guardar_imagen(os.path.join(destino, f"lfw_{i:04d}.jpg"), img_bgr)

print(f"Guardados {len(seleccion)} rostros en {destino}")
