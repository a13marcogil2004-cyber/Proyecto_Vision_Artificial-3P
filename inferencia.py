"""
inferencia.py
--------------
Script para correr el modelo YOLOv8 ya entrenado sobre imágenes nuevas
y generar evidencias con bounding boxes (detecciones de aves).

Uso:
    python scripts/inferencia.py --modelo modelo_aves_best.pt --imagenes ruta/a/imagenes --salida evidencias

Requiere que el modelo (best.pt) ya esté entrenado (ver notebooks/entrenamiento_yolo_aves.ipynb).
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def correr_inferencia(ruta_modelo: str, ruta_imagenes: str, ruta_salida: str, confianza: float = 0.35):
    """
    Corre el modelo YOLO sobre un conjunto de imágenes y guarda los resultados
    (imágenes con bounding boxes dibujados) en la carpeta de salida.

    Args:
        ruta_modelo: ruta al archivo .pt del modelo entrenado
        ruta_imagenes: carpeta con imágenes de prueba (jpg/png)
        ruta_salida: carpeta donde se guardarán las evidencias
        confianza: umbral mínimo de confianza para mostrar una detección
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{ruta_modelo}'. "
            "Asegúrate de haber entrenado primero con el notebook en notebooks/."
        )

    print(f"Cargando modelo desde: {ruta_modelo}")
    modelo = YOLO(ruta_modelo)

    os.makedirs(ruta_salida, exist_ok=True)

    print(f"Corriendo inferencia sobre imágenes en: {ruta_imagenes}")
    resultados = modelo.predict(
        source=ruta_imagenes,
        conf=confianza,
        save=True,
        project=ruta_salida,
        name="predicciones",
        exist_ok=True,
    )

    total_detecciones = sum(len(r.boxes) for r in resultados)
    print(f"\nListo. {len(resultados)} imágenes procesadas.")
    print(f"Total de aves detectadas: {total_detecciones}")
    print(f"Resultados guardados en: {Path(ruta_salida) / 'predicciones'}")

    return resultados


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de detección de aves con YOLOv8")
    parser.add_argument(
        "--modelo",
        type=str,
        default="modelo_aves_best.pt",
        help="Ruta al modelo entrenado (.pt)",
    )
    parser.add_argument(
        "--imagenes",
        type=str,
        required=True,
        help="Carpeta o ruta de imagen(es) de prueba",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="evidencias",
        help="Carpeta donde se guardarán las imágenes con detecciones",
    )
    parser.add_argument(
        "--confianza",
        type=float,
        default=0.35,
        help="Umbral mínimo de confianza (0 a 1) para considerar una detección válida",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    correr_inferencia(args.modelo, args.imagenes, args.salida, args.confianza)