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

MAX_LEN = 150

# def geom_collate(batch):
#     xs_list = [item[0] for item in batch]   # lista de tensores (T_i, d)
#     ys = torch.tensor([item[1] for item in batch], dtype=torch.long)

#     padded = []
#     for x in xs_list:
#         T, d = x.size()

#         # --- TRUNCATE ---
#         if T > MAX_LEN:
#             x = x[:MAX_LEN, :]   # truncar tiempo

#         # --- PAD ---
#         elif T < MAX_LEN:
#             pad_len = MAX_LEN - T
#             # pad en la dimensión temporal
#             # F.pad argumento: (pad_last_dim, pad_last_dim2, pad_first_dim, pad_first_dim2)
#             x = F.pad(x, (0, 0, 0, pad_len))  # padd al final en T

#         padded.append(x)

#     xs = torch.stack(padded, dim=0)  # (B, 150, d)
#     return xs, ys

# def geom_collate(batch):
#     """
#     Collate para HDM05WindowsDataset:
#     - x: (T, d_i) con d_i variable
#     - y: label
    
#     Solución: padding en la dimensión d para que todas sean (T, d_max).
#     """
#     xs_list = [item[0] for item in batch]   # cada x: (T, d_i)
#     ys = torch.tensor([item[1] for item in batch], dtype=torch.long)

#     # T es fijo. d_i cambia.
#     T = xs_list[0].size(0)
#     d_max = max(x.size(1) for x in xs_list)

#     padded = []
#     for x in xs_list:
#         d_i = x.size(1)
#         if d_i < d_max:
#             # pad dimensión de características
#             pad_amount = d_max - d_i
#             # pad: (left, right, top, bottom) pero aquí (0, pad_amount) en d
#             x = torch.nn.functional.pad(x, (0, pad_amount))
#         padded.append(x)

#     xs = torch.stack(padded, dim=0)   # (B, T, d_max)
#     return xs, ys


# ---------------------------------------------------------------------
# 2) Collate geométrico — devuelve tensor de matrices y tensor de labels
# ---------------------------------------------------------------------
# def geom_collate(batch):
#     xs_list = [item[0] for item in batch]   # lista de (T_i, d)
#     ys = torch.tensor([item[1] for item in batch], dtype=torch.long)

#     # lengths = [x.size(0) for x in xs_list]
#     T_max = max(x.size(0) for x in xs_list)
#     d = xs_list[0].size(1)

#     padded = []
#     for x in xs_list:
#         pad_len = T_max - x.size(0)
#         if pad_len > 0:
#             x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))  # pad time dimension
#         padded.append(x)

#     xs = torch.stack(padded, dim=0)  # (B, T_max, d)
#     return xs, ys

def geom_collate(batch):
    xs = torch.stack([item[0] for item in batch])     # U matrices
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return xs, ys

def graph_collate(batch):
    U_list = [item[0] for item in batch]
    A_list = [item[1] for item in batch]
    y_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return U_list, A_list, y_batch




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
        drop_last=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
