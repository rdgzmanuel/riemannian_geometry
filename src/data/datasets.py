from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from ..config.paths import HDM05_GRASSMANN_DIR, HDM05_WINDOWS_DIR, HDM05_SPD_DIR


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
            label = data["label"]

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
        root: Path = HDM05_SPD_DIR,
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
