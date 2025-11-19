# src/hdm05_grassmann/data/hdm05_loader.py
from pathlib import Path

import ezc3d
import numpy as np

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

    # print(c3d["parameters"]["POINT"]["LABELS"]["value"])

    labels = []
    for lbl in c3d["parameters"]["POINT"]["LABELS"]["value"]:
        split = lbl.strip().split(":")
        if len(split) == 2:
            labels.append(split[1])

    return xyz.astype(np.float32), labels


def compute_common_markers():
    """
    Escanea TODOS los C3D y devuelve:
        - sorted list de marcadores presentes en TODOS los ficheros
        - también imprime cuáles archivos tienen marcadores extra o missing
    """
    files = list_c3d_files(use_cuts=True, pattern="*.C3D")
    per_file = []   # [(path, markers, names), ...]

    marker_sets = []

    for p in files:
        markers, names = load_c3d_markers(p)
        per_file.append((p, markers, names))
        marker_sets.append(set(names))

    # intersección
    common = set.intersection(*marker_sets)

    return per_file, sorted(common)


def build_identity_joint_mapping(common_markers):
    """
    Construye automáticamente un joint_mapping de EXACTAMENTE 31 articulaciones
    seleccionadas desde los marcadores comunes reales del dataset.

    Selección basada en estabilidad + anatomía, validado contra HDM05.
    """

    # Los 31 marcadores seleccionados (ordenados y estables)
    selected_31 = [
        'C7',   # cuello
        'CLAV', # clavícula
        'LFHD', # cabeza
        'RBAC', # espalda alta
        'T10',  # torso
        'STRN', # sternum

        # LEFT ARM
        'LSHO', 'LELB', 'LFRM', 'LWRA', 'LWRB', 'LFIN',

        # RIGHT ARM
        'RSHO', 'RELB', 'RFRM', 'RWRA', 'RWRB',

        # LEFT LEG
        'LBWT', 'LFWT', 'LKNE', 'LANK', 'LHEE', 'LTOE', 'LMT5',

        # RIGHT LEG
        'RBWT', 'RFWT', 'RKNE', 'RANK', 'RHEE', 'RTOE', 'RMT5',
    ]

    # Validación: todos deben estar en los comunes
    missing = [m for m in selected_31 if m not in common_markers]
    if missing:
        raise ValueError(f"Los siguientes marcadores no están en los comunes: {missing}")

    # mapping identidad
    return {m: m for m in selected_31}


def markers_to_skeleton(
    markers: np.ndarray,
    marker_names: list[str],
    joint_mapping: dict[str, str],
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

    # T: frames
    # Jm: joint

    if joint_mapping is None:
        print("no joint mapping")
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
    markers: np.ndarray,
    marker_names: list[str],
    joint_mapping: dict[str, str]
) -> dict:
    """
    Carga un archivo .c3d y devuelve un dict con:
      - "skeleton": np.ndarray (T, J, 3)
      - "joint_names": list[str]
      - "fps": float
    """
    skel, joint_names = markers_to_skeleton(
        markers, marker_names, joint_mapping=joint_mapping
    )

    # FPS
    # c3d = ezc3d.c3d(str(path))
    # fps = float(c3d["header"]["points"]["frame_rate"])

    # return {"skeleton": skel, "joint_names": joint_names, "fps": fps}
    return {"skeleton": skel, "joint_names": joint_names}


if __name__ == "__main__":
    files = list_c3d_files(use_cuts=True, pattern="*.C3D")

    per_file, common_markers = compute_common_markers()
    joint_mapping = build_identity_joint_mapping(common_markers)

    print(joint_mapping)

    print(len(per_file))

    for p, markers, names in per_file:
        seq = load_sequence(markers, names, joint_mapping)

    print(len(seq["joint_names"]))
