"""
Descarga fotos variadas SIN personas conocidas (paisajes, ciudades, objetos...)
desde Picsum (imágenes libres de Unsplash) y las guarda en dataset/Fondo.

Uso:
    python descargar_fondos.py        # 100 imágenes
    python descargar_fondos.py 150
"""
import os
import sys
import urllib.request

import cv2
import numpy as np

from utils import guardar_imagen

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
destino = os.path.join("dataset", "Fondo")
os.makedirs(destino, exist_ok=True)

ok = 0
for i in range(1, N + 1):
    url = f"https://picsum.photos/seed/fondo{i}/400/300"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "actividad8-script"})
        datos = urllib.request.urlopen(req, timeout=30).read()
        img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and guardar_imagen(os.path.join(destino, f"picsum_{i:04d}.jpg"), img):
            ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"  falló {i}: {e}")
    if i % 20 == 0:
        print(f"{i}/{N}")

print(f"Guardadas {ok} imágenes en {destino}")
