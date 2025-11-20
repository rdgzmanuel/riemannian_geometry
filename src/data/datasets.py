from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from ..config.paths import HDM05_GRASSMANN_DIR, HDM05_WINDOWS_DIR, HDM05_SPD_DIR

def build_label_mapping(labels: list[str]) -> dict[str, int]:
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}


class HDM05WindowsDataset(Dataset):
    """
    Dataset Euclídeo de ventanas (T, d).
    Cada archivo tiene:
        windows: (Nw, T, d)
        label: str
        file_id: str
    """
    def __init__(
        self,
        root: Path = HDM05_WINDOWS_DIR,
        mapping_json: str ="./action2idx.json",
        split_filter: Callable[[str], bool] | None = None,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform

        # --- cargar mapping fijo ---
        with open(mapping_json, "r") as f:
            self.label2idx = json.load(f)

        # Lo invertimos por si necesitas recuperarlo
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        all_files = sorted(self.root.glob("*.npz"))

        self.items = []
        dims: list[str] = []

        # --- Escaneo inicial ---
        for f in all_files:
            data = np.load(f, allow_pickle=True)
            label = str(data["label"])
            file_id = str(data["file_id"])

            if split_filter and not split_filter(file_id):
                continue

            windows = data["windows"]  # (Nw, T, d)
            Nw, _, d = windows.shape

            dims.append(d)

            for i in range(Nw):
                self.items.append((f, i, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, win_idx, label_int = self.items[idx]
        data = np.load(path, allow_pickle=True)
        x = torch.from_numpy(data["windows"][win_idx]).float()
        y = torch.tensor(label_int).long()

        if self.transform:
            x = self.transform(x)

        return x, y


class HDM05SPDDataset(Dataset):
    def __init__(
        self,
        root: Path = HDM05_WINDOWS_DIR,
        mapping_json: str ="./action2idx.json",
        split_filter: Callable[[str], bool] | None = None,
        transform: Callable | None = None,
        key: str = "spds",
    ):
        self.root = Path(root)
        self.transform = transform
        self.key = key

        # mapping cargado del JSON
        with open(mapping_json, "r") as f:
            self.label2idx = json.load(f)
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        self.items = []
        dims = []

        for f in sorted(self.root.glob("*.npz")):
            data = np.load(f, allow_pickle=True)
            file_id = str(data["file_id"])
            label_int = int(data["label"])

            if split_filter and not split_filter(file_id):
                continue

            spds = data[self.key]
            Nw, d, _ = spds.shape
            dims.append(d)

            for i in range(Nw):
                self.items.append((f, i, label_int))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, spd_idx, label_int = self.items[idx]
        data = np.load(path, allow_pickle=True)

        X = torch.from_numpy(data[self.key][spd_idx]).float()  # (d, d)
        y = torch.tensor(label_int).long()

        if self.transform:
            X = self.transform(X)

        return X, y


# class HDM05GrassmannDataset(Dataset):
#     """
#     Dataset para GrNet / modelos en Grassmann:
#       - subspaces: (d, p) como matrices U
#       - con control opcional del número máximo de nodos/ventanas por archivo
#     """

#     def __init__(
#         self,
#         root: Path = HDM05_GRASSMANN_DIR,
#         split_filter: Callable[[str], bool] | None = None,
#         transform: Callable | None = None,
#         max_nodes: int | None = None,   # NUEVO
#         sampling: str = "uniform",      # "uniform" | "first"
#     ):
#         self.root = Path(root)
#         self.transform = transform
#         self.max_nodes = max_nodes
#         self.sampling = sampling

#         self.files = sorted(self.root.glob("*.npz"))
#         self.items: list[tuple[Path, int]] = []
#         labels: list[str] = []

#         self.max_d = 0

#         for f in self.files:
#             data = np.load(f, allow_pickle=True)
#             label = str(data["label"])
#             file_id = str(data["file_id"])

#             if split_filter is not None and not split_filter(file_id):
#                 continue

#             subs = data["subspaces"]  # (Nw, d, p)
#             Nw, d_i, p_i = subs.shape
#             if d_i > self.max_d:
#                 self.max_d = d_i

#             # ---------- NUEVO: controlar max_nodes ----------
#             if self.max_nodes is not None and Nw > self.max_nodes:
#                 if self.sampling == "first":
#                     idxs = list(range(self.max_nodes))
#                 else:
#                     # muestreo uniforme
#                     idxs = np.linspace(0, Nw - 1, self.max_nodes).astype(int)
#             else:
#                 idxs = list(range(Nw))
#             # -------------------------------------------------

#             # Añadir elementos al índice global
#             for i in idxs:
#                 self.items.append((f, i))
#                 labels.append(label)

#         self.label2idx = build_label_mapping(labels)

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, idx: int):
#         npz_path, sub_idx = self.items[idx]
#         data = np.load(npz_path, allow_pickle=True)
#         subs = data["subspaces"]  # (Nw, d, p)
#         label = str(data["label"])

#         U = subs[sub_idx]  # (d, p)
#         y = self.label2idx[label]

#         U = torch.from_numpy(U).float()

#         # padding filas a max_d
#         d_i, p = U.shape
#         if d_i < self.max_d:
#             pad_rows = self.max_d - d_i
#             U = torch.nn.functional.pad(U, (0, 0, 0, pad_rows))

#         if self.transform:
#             U = self.transform(U)

#         return U, torch.tensor(y).long()


class HDM05GrassmannDataset(Dataset):
    """
    Dataset para GrNet / modelos en Grassmann:
      - subspaces: (d, p) como matrices U
    """

    def __init__(
        self,
        root: Path = HDM05_GRASSMANN_DIR,
        split_filter: Callable[[str], bool] | None = None,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform

        self.files = sorted(self.root.glob("*.npz"))
        self.items: list[tuple[Path, int]] = []
        labels: list[str] = []

        self.max_d = 0

        for f in self.files:
            data = np.load(f, allow_pickle=True)
            label = str(data["label"])
            file_id = str(data["file_id"])

            if split_filter is not None and not split_filter(file_id):
                continue

            subs = data["subspaces"]  # (Nw, d, p)
            Nw, d_i, p_i = subs.shape
            if d_i > self.max_d:
                self.max_d = d_i   # nos quedamos con el mayor d

            for i in range(Nw):
                self.items.append((f, i))
                labels.append(label)

        self.label2idx = build_label_mapping(labels)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        npz_path, sub_idx = self.items[idx]
        data = np.load(npz_path, allow_pickle=True)
        subs = data["subspaces"]  # (Nw, d, p)
        label = str(data["label"])

        U = subs[sub_idx]  # (d, p)
        y = self.label2idx[label]

        U = torch.from_numpy(U).float()

        # padding de filas hasta max_d
        # print("U", type(U))
        d_i, p = U.shape
        if d_i < self.max_d:
            pad_rows = self.max_d - d_i
            # pad: (left, right, top, bottom) sobre (d, p)
            U = torch.nn.functional.pad(U, (0, 0, 0, pad_rows))  # (max_d, p)

        if self.transform is not None:
            U = self.transform(U)

        y = torch.tensor(y).long()

        return U, y