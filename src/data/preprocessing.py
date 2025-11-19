# src/hdm05_grassmann/data/preprocessing.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config.paths import HDM05_CUTS_C3D_DIR, INTERIM_DIR
from .hdm05_loader import (
    list_c3d_files,
    load_c3d_markers,
    load_sequence,
    build_identity_joint_mapping,
    compute_common_markers,
)


def center_skeleton(skel: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """
    Resta al esqueleto la posición del root_joint en cada frame.
    skel: (T, J, 3)
    """
    root = skel[:, root_joint : root_joint + 1, :]  # (T,1,3)
    return skel - root


def normalize_scale(skel: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Escalamos el esqueleto para que la desviación estándar global ≈ 1.
    """
    std = skel.reshape(-1, 3).std()
    if std < eps:
        return skel
    return skel / std


def remove_nan_frames(skel: np.ndarray) -> np.ndarray:
    """
    Elimina frames que contengan NaN en algún joint.
    """
    mask = ~np.isnan(skel).any(axis=(1, 2))
    return skel[mask]


def preprocess_sequence(
    seq: dict,
    root_joint: int = 0,
) -> dict:
    skel = seq["skeleton"]  # (T, J, 3)

    skel = remove_nan_frames(skel)
    if skel.shape[0] == 0:
        raise ValueError("Sequence has no valid frames after removing NaNs")

    skel = center_skeleton(skel, root_joint=root_joint)
    skel = normalize_scale(skel)

    return {
        "skeleton": skel,
        "joint_names": seq["joint_names"],
        # "fps": seq["fps"],
    }


def preprocess_all(
    src_dir: Path = HDM05_CUTS_C3D_DIR,
    dst_dir: Path = INTERIM_DIR,
    root_joint: int = 0,
):
    """
    Recorre todos los .c3d y salva npz limpios en INTERIM_DIR.
    """
    if src_dir == HDM05_CUTS_C3D_DIR:
        files = list_c3d_files(use_cuts=True, pattern="*.C3D")
    else:
        files = list_c3d_files(use_cuts=False, pattern=".*c3d")

    dst_dir.mkdir(parents=True, exist_ok=True)

    per_file, common_markers = compute_common_markers()
    joint_mapping = build_identity_joint_mapping(common_markers)

    # for path in files:
    for path, markers, names in per_file:
        print(f"Preprocessing {path.name}")
        markers, marker_names = load_c3d_markers(path)
        seq = load_sequence(markers, marker_names, joint_mapping)
        try:
            clean = preprocess_sequence(seq, root_joint=root_joint)
        except ValueError as e:
            print(f"  Skipping {path.name}: {e}")
            continue

        out_path = dst_dir / (path.stem + ".npz")
        np.savez_compressed(
            out_path,
            skeleton=clean["skeleton"],
            joint_names=np.array(clean["joint_names"]),
            # fps=clean["fps"],
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    preprocess_all()
