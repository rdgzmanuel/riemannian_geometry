from __future__ import annotations

from pathlib import Path
import numpy as np

from src.config.paths import HDM05_WINDOWS_DIR, HDM05_SPD_DIR


# -----------------------------------------------------------
# 1) COVARIANZA SPD
# -----------------------------------------------------------
def covariance_from_window(win: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    win: (T, d)
    Devuelve matriz SPD (d, d) = cov + eps*I
    """
    X = win - win.mean(axis=0, keepdims=True)
    T = X.shape[0]
    C = (X.T @ X) / max(T - 1, 1)
    C += eps * np.eye(C.shape[0], dtype=C.dtype)
    return C.astype(np.float32)


# -----------------------------------------------------------
# 2) CONVERTIR UNA VENTANA A SPD
# -----------------------------------------------------------
def window_to_spd(win: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Devuelve una matriz SPD (d, d).
    """
    return covariance_from_window(win, eps=eps)


# -----------------------------------------------------------
# 3) GENERAR SPD PARA UN ARCHIVO
# -----------------------------------------------------------
def build_spd_for_file(
    npz_path: Path,
    eps: float = 1e-6,
) -> tuple[np.ndarray, str]:
    """
    Carga un npz de ventanas y devuelve:
      - spds: (Nw, d, d)
      - label: str
    """
    data = np.load(npz_path, allow_pickle=True)
    windows = data["windows"]  # (Nw, T, d)
    label = str(data["label"])

    Nw, T, d = windows.shape
    spds = np.zeros((Nw, d, d), dtype=np.float32)

    for i in range(Nw):
        spds[i] = window_to_spd(windows[i], eps=eps)

    return spds, label


# -----------------------------------------------------------
# 4) GENERAR SPD PARA TODO HDM05
# -----------------------------------------------------------
def build_all_spd(
    src_dir: Path = HDM05_WINDOWS_DIR,
    dst_dir: Path = HDM05_SPD_DIR,
    eps: float = 1e-6,
):
    """
    Recorre todos los npz de ventanas y genera:
      - spds: (Nw, d, d)
      - label: str
      - file_id: str
    (equivalente a build_all_grassmann pero para SPD)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(src_dir.glob("*.npz"))

    for npz_path in npz_files:
        print(f"SPD repr {npz_path.name}")
        spds, label = build_spd_for_file(npz_path, eps=eps)

        if spds.shape[0] == 0:
            print("  No windows, skipping")
            continue

        data = np.load(npz_path, allow_pickle=True)
        file_id = str(data["file_id"])

        out_path = dst_dir / npz_path.name
        np.savez_compressed(
            out_path,
            spds=spds,
            label=label,
            file_id=file_id,
        )

        print(f" Saved {out_path}")
