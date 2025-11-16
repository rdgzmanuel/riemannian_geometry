# src/hdm05_grassmann/data/hdm05_loader.py
from pathlib import Path

import ezc3d
import numpy as np

# DATA_DIR = Path("data")
# RAW_DIR = DATA_DIR / "HDM05"
# HDM05_FULL_C3D_DIR = RAW_DIR / "full_takes"
# HDM05_CUTS_C3D_DIR = RAW_DIR / "cuts"
from ..config.paths import HDM05_CUTS_C3D_DIR, HDM05_FULL_C3D_DIR


def list_c3d_files(use_cuts: bool = True, pattern: str = "*.C3D") -> list[Path]:
    base = HDM05_CUTS_C3D_DIR if use_cuts else HDM05_FULL_C3D_DIR
    return sorted(base.glob(pattern))


def load_c3d_markers(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Carga un .c3d de HDM05 y devuelve:
      - markers: (T, J, 3) en mm
      - marker_names: lista de nombres de marcadores (longitud J)
    """
    c3d = ezc3d.c3d(str(path))
    points = c3d["data"]["points"]  # shape: (4, J, T), 4 = (x,y,z,1/flag)
    xyz = points[:3, :, :]  # (3, J, T)
    xyz = np.transpose(xyz, (2, 1, 0))  # (T, J, 3)

    labels = [lbl.strip() for lbl in c3d["parameters"]["POINT"]["LABELS"]["value"]]

    return xyz.astype(np.float32), labels


def markers_to_skeleton(
    markers: np.ndarray,
    marker_names: list[str],
    joint_mapping: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Convierte markers HDM05 → "joints" de esqueleto.

    markers: (T, Jm, 3)
    marker_names: nombres de los Jm marcadores

    joint_mapping: dict opcional:
        {joint_name: marker_name}
    Si es None, usamos todos los marcadores tal cual.
    """
    T, Jm, _ = markers.shape

    if joint_mapping is None:
        # usar todos los marcadores como "joints"
        return markers, marker_names

    joint_names = list(joint_mapping.keys())
    J = len(joint_names)
    skel = np.zeros((T, J, 3), dtype=np.float32)

    name_to_idx = {n: i for i, n in enumerate(marker_names)}

    for j, jname in enumerate(joint_names):
        mname = joint_mapping[jname]
        if mname not in name_to_idx:
            raise ValueError(f"Marker {mname} not found in C3D file")
        mi = name_to_idx[mname]
        skel[:, j, :] = markers[:, mi, :]

    return skel, joint_names


def load_sequence(
    path: Path,
    joint_mapping: dict[str, str] | None = None,
) -> dict:
    """
    Carga un archivo .c3d y devuelve un dict con:
      - "skeleton": np.ndarray (T, J, 3)
      - "joint_names": list[str]
      - "fps": float
    """
    markers, marker_names = load_c3d_markers(path)

    skel, joint_names = markers_to_skeleton(
        markers, marker_names, joint_mapping=joint_mapping
    )

    # FPS
    c3d = ezc3d.c3d(str(path))
    fps = float(c3d["header"]["points"]["frame_rate"])

    return {"skeleton": skel, "joint_names": joint_names}
