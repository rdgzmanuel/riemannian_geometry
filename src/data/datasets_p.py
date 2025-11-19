from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
    Dataset euclídeo: ventanas (T, d) + label.
    Bueno como baseline MLP/LSTM.
    """

    def __init__(
        self,
        root: Path = HDM05_WINDOWS_DIR,
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

            windows = data["windows"]  # (Nw, T, d_i)
            d_i = windows.shape[2]
            if d_i > self.max_d:
                self.max_d = d_i

            # print(">>> HDM05WindowsDataset max_d =", self.max_d)

            if split_filter is not None and not split_filter(file_id):
                continue

            Nw = windows.shape[0]
            # guardamos un indice por ventana
            for i in range(Nw):
                self.items.append((f, i))
                labels.append(label)

        self.label2idx = build_label_mapping(labels)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        npz_path, win_idx = self.items[idx]
        data = np.load(npz_path, allow_pickle=True)
        windows = data["windows"]  # (Nw, T, d)
        label = str(data["label"])

        d_i = windows.shape[2]
        if d_i > self.max_d:
            self.max_d = d_i
        x = windows[win_idx]  # (T, d)
        y = self.label2idx[label]

        x = torch.from_numpy(x).float()  # (T, d)
        y = torch.tensor(y).long()

        # print("type x:", type(x), "shape U:", x.shape)
        # print("type y:", type(y), "value y:", y.item())
        if self.transform is not None:
            x = self.transform(x)

        T, d_i = x.shape
        if d_i < self.max_d:
            pad_d = self.max_d - d_i
            x = torch.nn.functional.pad(x, (0, pad_d))  # pad feature dim

        return x, y
    

class HDM05SPDDataset(Dataset):
    """
    Dataset para modelos SPD:
      Cada item es una matriz SPD (d,d) + etiqueta.
      Se hace padding para igualar el tamaño máximo.
    """

    def __init__(
        self,
        root: Path = HDM05_SPD_DIR,
        split_filter: Callable[[str], bool] | None = None,
        transform: Callable | None = None,
        key: str = "spds"
    ):
        """
        root: carpeta con npz que contienen matrices SPD
        key: clave dentro del npz (por defecto 'spds'),
             p.ej. stored['spds'] -> (Nw, d, d)
        """
        self.root = Path(root)
        self.transform = transform
        self.key = key

        self.files = sorted(self.root.glob("*.npz"))
        self.items: list[tuple[Path, int]] = []
        labels: list[str] = []

        self.max_d = 0

        # Escaneamos todos los ficheros y puntos
        for f in self.files:
            data = np.load(f, allow_pickle=True)
            label = str(data["label"])
            file_id = str(data["file_id"])

            if split_filter is not None and not split_filter(file_id):
                continue

            spds = data[key]          # (Nw, d_i, d_i)
            Nw, d_i, _ = spds.shape

            # actualizamos el tamaño máximo
            if d_i > self.max_d:
                self.max_d = d_i

            # cada ventana se guarda como item individual
            for i in range(Nw):
                self.items.append((f, i))
                labels.append(label)

        # asignamos mapping de labels
        self.label2idx = build_label_mapping(labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npz_path, spd_idx = self.items[idx]

        data = np.load(npz_path, allow_pickle=True)
        spds = data[self.key]        # (Nw, d, d)
        label = str(data["label"])

        X = spds[spd_idx]            # (d, d), SPD
        X = torch.from_numpy(X).float()

        d_i = X.shape[0]

        # padding para llegar a max_d
        if d_i < self.max_d:
            pad = self.max_d - d_i
            # pad: (left, right, top, bottom)
            X = torch.nn.functional.pad(X, (0, pad, 0, pad))

        if self.transform is not None:
            X = self.transform(X)

        y = torch.tensor(self.label2idx[label], dtype=torch.long)

        return X, y


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


class HDM05GrassmannGraphDataset(Dataset):
    """
    Dataset de grafos:
      - Agrupa ventanas del mismo file_id en un grafo.
      - Cada nodo: U ∈ R^{d×p}.
      - Adyacencia temporal simple.
    """

    def __init__(self, base_ds: HDM05GrassmannDataset):
        super().__init__()
        self.base_ds = base_ds

        # agrupar por file_id leyendo el npz original
        groups = defaultdict(list)
        labels_by_file = {}

        for idx, (npz_path, win_idx) in enumerate(base_ds.items):
            data = np.load(npz_path, allow_pickle=True)
            file_id = str(data["file_id"])
            label = str(data["label"])
            groups[file_id].append(idx)
            labels_by_file[file_id] = label

        self.file_ids = list(groups.keys())
        self.groups = groups
        self.labels_by_file = labels_by_file

        # label2idx común
        all_labels = list(labels_by_file.values())
        self.label2idx = {lbl: i for i, lbl in enumerate(sorted(set(all_labels)))}

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # print([U.shape for U, _ in (self.base_ds[si] for si in range(len(self.base_ds)))])

        file_id = self.file_ids[idx]
        idxs = self.groups[file_id]

        U_list = []
        for si in idxs:
            U, _ = self.base_ds[si]  # (d, p)
            U_list.append(U)

        U = torch.stack(U_list, dim=0)  # (N, d, p)
        N, d, p = U.shape

        A = torch.eye(N)
        if N > 1:
            A[:-1, 1:] = torch.eye(N - 1)
            A[1:, :-1] = torch.eye(N - 1)

        y = self.label2idx[self.labels_by_file[file_id]]
        y = torch.tensor(y, dtype=torch.long)

        return U, A, y
