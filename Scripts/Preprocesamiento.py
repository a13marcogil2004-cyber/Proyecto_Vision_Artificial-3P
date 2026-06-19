"""
preprocesamiento.py
--------------------
Script de validación y preprocesamiento del dataset antes de entrenar.

Verifica que:
- La estructura de carpetas sea la esperada por YOLO (train/valid/test con images/ y labels/)
- Cada imagen tenga su archivo de etiquetas correspondiente
- Las etiquetas tengan el formato correcto (class x_center y_center width height, valores 0-1)
- Reporta estadísticas básicas del dataset (cantidad de imágenes, cantidad de cajas, etc.)

Uso:
    python scripts/preprocesamiento.py --dataset ruta/al/dataset
"""

import argparse
import os
from pathlib import Path

import yaml


def validar_split(ruta_split: str, nombre_split: str) -> dict:
    """Valida un split (train/valid/test) y devuelve estadísticas."""
    carpeta_imagenes = Path(ruta_split) / "images"
    carpeta_labels = Path(ruta_split) / "labels"

    stats = {
        "split": nombre_split,
        "imagenes": 0,
        "labels": 0,
        "imagenes_sin_label": [],
        "labels_mal_formateados": [],
        "total_cajas": 0,
    }

    if not carpeta_imagenes.exists():
        print(f"  [!] No existe la carpeta de imágenes: {carpeta_imagenes}")
        return stats

    extensiones_validas = {".jpg", ".jpeg", ".png"}
    imagenes = [f for f in carpeta_imagenes.iterdir() if f.suffix.lower() in extensiones_validas]
    stats["imagenes"] = len(imagenes)

    for img_path in imagenes:
        label_path = carpeta_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats["imagenes_sin_label"].append(img_path.name)
            continue

        stats["labels"] += 1
        with open(label_path, "r") as f:
            lineas = [l.strip() for l in f.readlines() if l.strip()]
