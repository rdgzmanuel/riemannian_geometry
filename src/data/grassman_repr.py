# src/hdm05_grassmann/data/grassmann_repr.py
from __future__ import annotations

from pathlib import Path

import numpy as np

# DATA_DIR = Path("data")
# PROCESSED_DIR = DATA_DIR / "processed"
# HDM05_WINDOWS_DIR = PROCESSED_DIR / "hdm05_windows"
# HDM05_GRASSMANN_DIR = PROCESSED_DIR / "hdm05_grassmann"
from ..config.paths import HDM05_GRASSMANN_DIR, HDM05_WINDOWS_DIR


def covariance_from_window(win: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    win: (T, d)
    Devuelve matriz de covarianza (d, d)
    """
    X = win - win.mean(axis=0, keepdims=True)
    T = X.shape[0]
    C = (X.T @ X) / max(T - 1, 1)
    C += eps * np.eye(C.shape[0], dtype=C.dtype)
    return C


def window_to_grassmann(
    win: np.ndarray,
    p: int,
) -> np.ndarray:
    """
    win: (T, d)
    Devuelve U ∈ R^{d×p} con columnas ortonormales (p subespacio en Gr(d,p)).
    """
    C = covariance_from_window(win)  # (d, d)
    # SVD
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    U_p = U[:, :p]
    return U_p.astype(np.float32)


def build_grassmann_for_file(
    npz_path: Path,
    p: int,
) -> tuple[np.ndarray, str]:
    """
    Carga un npz de ventanas y devuelve:
      - subspaces: (Nw, d, p)
      - label: str
    """
    data = np.load(npz_path, allow_pickle=True)
    windows = data["windows"]  # (Nw, T, d)
    label_int = int(data["label"])

    Nw, T, d = windows.shape
    subspaces = np.zeros((Nw, d, p), dtype=np.float32)

    for i in range(Nw):
        subspaces[i] = window_to_grassmann(windows[i], p=p)

    return subspaces, label_int


def build_all_grassmann(
    src_dir: Path = HDM05_WINDOWS_DIR,
    dst_dir: Path = HDM05_GRASSMANN_DIR,
    p: int = 10,
):
    """
    Recorre todos los npz de windows y guarda:
      - subspaces: (Nw, d, p)
      - label: str
      - file_id: str
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(src_dir.glob("*.npz"))

    for npz_path in npz_files:
        print(f"Grassmann repr {npz_path.name}")
        subspaces, label_int = build_grassmann_for_file(npz_path, p=p)
        if subspaces.shape[0] == 0:
            print("  No windows, skipping")
            continue

        data = np.load(npz_path, allow_pickle=True)
        file_id = str(data["file_id"])

        out_path = dst_dir / npz_path.name
        np.savez_compressed(
            out_path,
            subspaces=subspaces,
            label=label_int,
            file_id=file_id,
        )
        print(f" Saved {out_path}")
