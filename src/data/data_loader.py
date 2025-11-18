import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random


# ---------------------------------------------------------------------
# 1) Reproducibilidad total
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def geom_collate(batch):
    xs = torch.stack([item[0] for item in batch])     # U matrices
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return xs, ys

def graph_collate(batch):
    """
    batch = [(U, A, y), ...]
      - U: (N_i, d, p)  (N_i = nº nodos del grafo i)
      - A: (N_i, N_i)
      - y: escalar

    Devolvemos:
      - U_batch: (B, N_max, d, p)
      - A_batch: (B, N_max, N_max)
      - y_batch: (B,)
    con padding de nodos (nodos “dummy” sin conexiones).
    """
    U_list = [item[0] for item in batch]
    A_list = [item[1] for item in batch]
    y_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    # tamaños base
    N_max = max(U.shape[0] for U in U_list)
    d = U_list[0].shape[1]
    p = U_list[0].shape[2]

    U_padded = []
    A_padded = []

    for U, A in zip(U_list, A_list):
        N = U.shape[0]
        # pad nodos si hace falta
        if N < N_max:
            pad_nodes = N_max - N

            # U: (N, d, p) -> (N_max, d, p)
            # pad: (pad_last_dim_p0, pad_last_dim_p1, pad_dim1_0, pad_dim1_1, pad_dim0_0, pad_dim0_1)
            U = F.pad(U, (0, 0, 0, 0, 0, pad_nodes))

            # A: (N, N) -> (N_max, N_max)
            # pad: (left, right, top, bottom) en 2D
            A = F.pad(A, (0, pad_nodes, 0, pad_nodes))

        U_padded.append(U)
        A_padded.append(A)

    U_batch = torch.stack(U_padded, dim=0)  # (B, N_max, d, p)
    A_batch = torch.stack(A_padded, dim=0)  # (B, N_max, N_max)

    return U_batch, A_batch, y_batch




# ---------------------------------------------------------------------
# 3) Función general para splits + loaders
# ---------------------------------------------------------------------
def get_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    collate_fn=geom_collate):

    set_seed(seed)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,     # prueba 2, 4, 8 según tu CPU
        pin_memory=True,   # ayuda a copiar más rápido a la GPU
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
