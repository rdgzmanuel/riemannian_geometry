# src/hdm05_grassmann/data/windowing.py
from __future__ import annotations

from pathlib import Path

import numpy as np

# DATA_DIR = Path("data")
# RAW_DIR = DATA_DIR / "HDM05"
# INTERIM_DIR = DATA_DIR / "interim" / "hdm05_cleaned"
# PROCESSED_DIR = DATA_DIR / "processed"
# HDM05_WINDOWS_DIR = PROCESSED_DIR / "hdm05_windows"
from ..config.paths import HDM05_WINDOWS_DIR, INTERIM_DIR

import re

def extract_clean_label(npz_path: Path) -> str:
    """
    A partir de un archivo tipo:
      HDM_bd_<CLASE><sufijo>_<rep>_<fps>.C3D
    devuelve la clase limpia:
      - Elimina "Reps", "hops"...
      - Elimina números finales
      - Mantiene números internos del nombre
    """

    stem = npz_path.stem                     # HDM_bd_cartwheelLHandStart1Reps_003_120
    parts = stem.split("_")
    raw = parts[2]                           # "cartwheelLHandStart1Reps"

    # 1) quitar sufijo "Reps", si existe
    raw = re.sub(r"Reps$", "", raw)

    # 2) quitar sufijo que empieza en un número seguido de letras (ej: "3hops")
    raw = re.sub(r"\d+[A-Za-z]+$", "", raw)

    # 3) quitar número final simple (ej: "1" en "Start1")
    raw = re.sub(r"\d+$", "", raw)

    return raw


def skeleton_to_flat(
    skel: np.ndarray,
) -> np.ndarray:
    """
    skel: (T, J, 3) → (T, J*3)
    """
    T, J, C = skel.shape
    assert C == 3
    return skel.reshape(T, J * C)


def sliding_windows(
    seq: np.ndarray,
    window_size: int,
    stride: int,
    min_frames: int | None = None,
) -> list[np.ndarray]:
    """
    seq: (T, d)
    Devuelve lista de ventanas (window_size, d).
    """
    T = seq.shape[0]
    if min_frames is None:
        min_frames = window_size

    if min_frames > T:
        return []

    windows = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(seq[start:end])

    return windows


def build_windows_for_file(
    npz_path: Path,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, str]:
    """
    Carga un .npz del INTERIM y devuelve:
      - windows: (Nw, T, d)
      - label: string con la clase (de momento, parseamos del nombre)
    """
    data = np.load(npz_path, allow_pickle=True)
    skel = data["skeleton"]  # (T, J, 3)
    flat = skeleton_to_flat(skel)  # (T, d)

    wins = sliding_windows(flat, window_size=window_size, stride=stride)
    if not wins:
        return np.empty((0, window_size, flat.shape[1]), dtype=np.float32), ""

    windows = np.stack(wins).astype(np.float32)

    # Ejemplo: nombre tipo "cartwheel_r01.npz" → clase "cartwheel"
    # label = npz_path.stem.split("_")[0]
    label = extract_clean_label(npz_path)

    return windows, label


def build_all_windows(
    src_dir: Path = INTERIM_DIR,
    dst_dir: Path = HDM05_WINDOWS_DIR,
    window_size: int = 32,
    stride: int = 16,
):
    """
    Recorre todos los .npz de INTERIM y guarda ventanas en PROCESSED/hdm05_windows.

    Formato de salida:
      - un npz por secuencia original:
        * windows: (Nw, T, d)
        * label: str
        * file_id: str
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(src_dir.glob("*.npz"))

    for npz_path in npz_files:
        print(f"Windowing {npz_path.name}")
        windows, label = build_windows_for_file(
            npz_path, window_size=window_size, stride=stride
        )
        if windows.shape[0] == 0:
            print("  No windows, skipping")
            continue

        out_path = dst_dir / npz_path.name
        np.savez_compressed(
            out_path,
            windows=windows,
            label=label,
            file_id=npz_path.stem,
        )
        print(f"Saved {out_path}")
