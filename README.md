# Actividad 8: Clasificación Estudiante vs Fondo con VGG-16

Clasificador de imágenes con **transfer learning (VGG-16)** que reconoce el rostro de un
estudiante (`Joel`) y lo distingue de cualquier otra cosa (`Fondo`: lugares, objetos y
rostros de otras personas). Se presenta con una app de **Streamlit**.

## Pruebas de la actividad

| Prueba | Imagen de entrada | Salida esperada |
|---|---|---|
| 1 | Foto del estudiante con la webcam | `Joel` |
| 2 | Fondo descargado de Google Images | `Fondo` |
| 3 | Celebridad parecida al estudiante | `Fondo` |

## Archivos

| Archivo | Descripción |
|---|---|
| `Actividad_8_Rostro_vs_Fondo_VGG16.ipynb` | Notebook de entrenamiento y evaluación (con resultados) |
| `app.py` | App de Streamlit para la presentación (webcam o subir imagen) |
| `utils.py` | Preprocesamiento compartido: recorte de rostro, RGB, normalización VGG-16 |
| `capturar_fotos.py` | Toma fotos con la webcam y las guarda en `dataset/<clase>` |
| `descargar_rostros.py` | Descarga rostros de otras personas (dataset LFW) a `dataset/Fondo` |
| `descargar_rostros_hispanos.py` | Agrega rostros LFW con nombres hispanos (para la prueba 3) |
| `descargar_fondos.py` | Descarga fotos variadas (Picsum) a `dataset/Fondo` |
| `Clasificacion_imagenes_perros_gatos_VGG-16_transfer_learning.ipynb` | Notebook base del curso |

## Mejoras respecto al notebook base

- **Recorte del rostro** (Haar cascade de OpenCV) para que el modelo aprenda la cara y no el fondo de la foto.
- **`preprocess_input` de VGG-16** y conversión **BGR → RGB**.
- **Aumento de datos**: giros, rotación, zoom, traslación, contraste y brillo.
- **Rostros de otras personas en `Fondo`** para que una celebridad parecida no se confunda con el estudiante.
- `GlobalAveragePooling2D`, pesos por clase y *fine-tuning* del bloque 5 de VGG-16.
- Umbral de confianza ajustable en la app.

**Lección aprendida:** con fotos de una sola sesión (misma luz y pared) el modelo daba 100% en
validación, pero no reconocía al estudiante en fotos nuevas: había aprendido el fondo, no la cara.
Agregar fotos de **3 sesiones distintas** (luz, pared y ropa diferentes) lo resolvió.

## Cómo ejecutarlo

Las imágenes y el modelo entrenado **no están en el repositorio** (son pesados y personales).

```bash
# 1. Entorno (Python 3.11)
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Dataset
python capturar_fotos.py Joel           # ≥150 fotos propias, en varias sesiones/luces/lugares
python capturar_fotos.py Fondo          # fotos del lugar sin la persona
python descargar_rostros.py 300         # rostros de otras personas
python descargar_rostros_hispanos.py    # rostros parecidos (prueba 3)
python descargar_fondos.py 120          # fondos variados

# 3. Entrenar: ejecutar el notebook completo -> genera modelo_rostro_fondo.keras y clases.json

# 4. App
streamlit run app.py
```

> En Windows, si TensorFlow falla con `DLL load failed`, instala el
> [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) más reciente.
