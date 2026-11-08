"""
Captura fotos con la webcam y las guarda en dataset/<carpeta>.

Uso:
    python capturar_fotos.py Joel        # fotos tuyas
    python capturar_fotos.py Fondo       # fotos de fondos (sin tu cara)

Controles:
    ESPACIO -> guarda una foto
    r       -> modo ráfaga on/off (guarda ~3 fotos por segundo)
    q / ESC -> salir

Consejo: muévete, gira la cabeza, cambia de expresión, de luz y de lugar,
ponte/quítate lentes o gorra. La variedad es más importante que la cantidad.
"""
import os
import sys
import time

import cv2

from utils import detectar_rostro, guardar_imagen

carpeta = sys.argv[1] if len(sys.argv) > 1 else "Joel"
destino = os.path.join("dataset", carpeta)
os.makedirs(destino, exist_ok=True)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    sys.exit("No se pudo abrir la webcam.")

contador = len(os.listdir(destino))
rafaga = False
ultima = 0.0
print(f"Guardando en {destino}. ESPACIO=foto, r=ráfaga, q=salir")

while True:
    ok, frame = cam.read()
    if not ok:
        break

    # Para fotos tuyas exigimos que se detecte el rostro; para "Fondo" no.
    es_fondo = carpeta.lower() == "fondo"
    caja = None if es_fondo else detectar_rostro(frame)
    rostro_ok = es_fondo or caja is not None

    guardar = False
    tecla = cv2.waitKey(1) & 0xFF
    if tecla in (ord("q"), 27):
        break
    if tecla == ord("r"):
        rafaga = not rafaga
    if tecla == ord(" "):
        guardar = True
    if rafaga and time.time() - ultima > 0.33:
        guardar = True

    if guardar and rostro_ok:
        contador += 1
        guardar_imagen(os.path.join(destino, f"{carpeta}_{contador:04d}.jpg"), frame)
        ultima = time.time()

    vista = frame.copy()
    if caja is not None:
        x, y, w, h = caja
        cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 255, 0), 2)
    texto = f"{carpeta}: {contador} fotos" + ("  [RAFAGA]" if rafaga else "")
    color = (0, 255, 0) if rostro_ok else (0, 0, 255)
    cv2.putText(vista, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if not rostro_ok:
        cv2.putText(vista, "Rostro no detectado: no se guarda", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Captura (q para salir)", vista)

cam.release()
cv2.destroyAllWindows()
print(f"Listo: {contador} fotos en {destino}")
