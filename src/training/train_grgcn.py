# src/hdm05_grassmann/training/train_grgcn.py

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from src.data.datasets import HDM05GrassmannDataset
from src.models.grgcn import GrGCNPlusPlusNetGeomstats
from torch.utils.data import DataLoader, Dataset, random_split

from .losses import get_classification_loss
from .utils import get_device, save_checkpoint, set_seed


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
        file_id = self.file_ids[idx]
        idxs = self.groups[file_id]

        U_list = []
        for si in idxs:
            U, _ = self.base_ds[si]  # (d, p)
            U_list.append(torch.tensor(U, dtype=torch.float32))

        U = torch.stack(U_list, dim=0)  # (N, d, p)
        N, d, p = U.shape

        A = torch.eye(N)
        if N > 1:
            A[:-1, 1:] = torch.eye(N - 1)
            A[1:, :-1] = torch.eye(N - 1)

        y = self.label2idx[self.labels_by_file[file_id]]
        y = torch.tensor(y, dtype=torch.long)

        return U, A, y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento GrGCN++ (Geomstats) en HDM05"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/grgcn/grgcn_geomstats.pt",
    )
    parser.add_argument("--gcn_layers", type=str, default="16,8")
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.5)
    return parser.parse_args()


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def train_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    n_samples = 0

    for U, A, y in loader:
        U = U.to(device)  # (B, N, d, p)
        A = A.to(device)  # (B, N, N)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(U, A)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        n_samples += y.size(0)

    return total_loss / n_samples, total_correct / n_samples


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    gcn_layers = parse_int_list(args.gcn_layers)
    hidden_dims = parse_int_list(args.hidden_dims)

    base_ds = HDM05GrassmannDataset()
    graph_ds = HDM05GrassmannGraphDataset(base_ds)

    n_total = len(graph_ds)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(graph_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    U0, A0, y0 = graph_ds[0]
    N0, d, p_in = U0.shape
    num_classes = len(graph_ds.label2idx)

    print(
        f"📊 GrGCN++ Geomstats: d={d}, p_in={p_in}, gcn_layers={gcn_layers}, num_classes={num_classes}"
    )

    model = GrGCNPlusPlusNetGeomstats(
        d=d,
        p_in=p_in,
        num_classes=num_classes,
        gcn_layers=gcn_layers,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = get_classification_loss("ce")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, device, criterion, optimizer
        )

        # Validación
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for U, A, y in val_loader:
                U = U.to(device)
                A = A.to(device)
                y = y.to(device)

                logits = model(U, A)
                loss = criterion(logits, y)

                total_loss += loss.item() * y.size(0)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += y.size(0)

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.checkpoint, model, optimizer, epoch, best_val_acc)

    print(f"🏁 Finalizado. Mejor val_acc={best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
