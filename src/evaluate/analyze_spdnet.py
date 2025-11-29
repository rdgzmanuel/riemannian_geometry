"""
ANÁLISIS COMPLETO DE SPDNet SOBRE HDM05
---------------------------------------

Genera:
 - Espectros de eigenvalores (BiMap1/2/3)
 - Condicionamiento por capa
 - Distribución de normas (LogEig)
 - t-SNE entrada vs salida
 - UMAP entrada vs salida (opcional)
 - Guarda todo en carpeta ./analysis_spdnet/

Requiere:
 - forward() corregido que devuelva {'bimap1','bimap2','bimap3','logeig'}
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from src.data.data_loader import get_dataloaders
from src.models.spdnet import SPDNet
from src.data.datasets import HDM05SPDDataset


# ============================================================
# 1. CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./analysis_spdnet"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. FIGURE HELPERS
# ============================================================

def savefig(name):
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, name)
    plt.savefig(path, dpi=200)
    print(f"[OK] Figura guardada: {path}")
    plt.close()


def plot_eig_distribution(mats, title, fname):
    eigs = []
    for M in mats:
        vals = np.linalg.eigvalsh(M)
        eigs.append(vals)
    eigs = np.array(eigs)

    plt.figure(figsize=(6, 4))
    plt.boxplot(eigs, showfliers=False)
    plt.title(title)
    plt.xlabel("Índice de autovalor")
    plt.ylabel("Valor")
    savefig(fname)


def plot_condition_numbers(mats_list, labels, fname):
    conds = []
    for mats in mats_list:
        L = []
        for M in mats:
            vals = np.linalg.eigvalsh(M)
            L.append(vals[-1] / vals[0])
        conds.append(L)

    plt.figure(figsize=(6,5))
    plt.boxplot(conds, labels=labels)
    plt.ylabel("Número de condicionamiento")
    plt.title("Condicionamiento por capa BiMap")
    savefig(fname)


def plot_norms(mats, fname):
    norms = [np.linalg.norm(M, 'fro') for M in mats]
    plt.figure(figsize=(6,4))
    plt.hist(norms, bins=30)
    plt.title("Distribución de normas Frobenius (LogEig)")
    plt.xlabel("Norma Frobenius")
    savefig(fname)


def plot_tsne(Z, labels, title, fname):
    Z2 = TSNE(n_components=2, perplexity=30).fit_transform(Z)
    plt.figure(figsize=(5,5))
    plt.scatter(Z2[:,0], Z2[:,1], c=labels, cmap="tab20", s=10)
    plt.title(title)
    plt.axis('off')
    savefig(fname)


def plot_umap(Z, labels, title, fname):
    reducer = umap.UMAP(n_components=2)
    Z2 = reducer.fit_transform(Z)

    plt.figure(figsize=(5,5))
    plt.scatter(Z2[:,0], Z2[:,1], c=labels, cmap="tab20", s=10)
    plt.title(title)
    plt.axis('off')
    savefig(fname)


# ============================================================
# 3. MAIN ANALYSIS
# ============================================================

def main():
    print(f"DEVICE = {DEVICE}")

    # Dataset y data loaders
    ds = HDM05SPDDataset()
    train_loader, val_loader, test_loader = get_dataloaders(
        ds,
        batch_size=32,
        seed=42
    )

    # Tomar un batch
    X, y = next(iter(val_loader))
    X = X.to(DEVICE)
    y = y.numpy()

    # Construir modelo
    d_in = X.size(1)
    num_classes = len(ds.label2idx)
    model = SPDNet(
        d_in=d_in,
        proj_dim=[70, 50, 30],
        num_classes=num_classes,
        debug=True
    ).to(DEVICE)

    # Cargar pesos entrenados
    ckpt = torch.load("experiments/checkpoints/spd/spdnet_geom.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Forward con extracción de features
    with torch.no_grad():
        _, feats = model(X)

    # ========================
    # PLOTS
    # ========================

    # 1. Autovalores por capa
    plot_eig_distribution(
        feats["bimap1"],
        "Espectro de autovalores tras BiMap1",
        "eigs_bimap1.png"
    )
    plot_eig_distribution(
        feats["bimap2"],
        "Espectro de autovalores tras BiMap2",
        "eigs_bimap2.png"
    )
    plot_eig_distribution(
        feats["bimap3"],
        "Espectro de autovalores tras BiMap3",
        "eigs_bimap3.png"
    )

    # 2. Condicionamiento de cada capa BiMap
    plot_condition_numbers(
        [feats["bimap1"], feats["bimap2"], feats["bimap3"]],
        ["BiMap1", "BiMap2", "BiMap3"],
        "conditioning_bimaps.png"
    )

    # 3. Normas Frobenius de LogEig
    plot_norms(
        feats["logeig"],
        "norms_logeig.png"
    )

    # 4. t-SNE de entrada vs salida
    Z_input = X.cpu().numpy().reshape(X.size(0), -1)
    Z_output = feats["logeig"].reshape(X.size(0), -1).numpy()

    plot_tsne(Z_input, y, "t-SNE Entrada (matrices originales)", "tsne_input.png")
    plot_tsne(Z_output, y, "t-SNE Salida SPDNet (LogEig)", "tsne_output.png")

    # 5. UMAP (si disponible)
    if HAS_UMAP:
        plot_umap(Z_input, y, "UMAP Entrada", "umap_input.png")
        plot_umap(Z_output, y, "UMAP Salida SPDNet", "umap_output.png")
    else:
        print("UMAP no está instalado. Saltando estas figuras.")

    print("\n✔ TODAS LAS FIGURAS GENERADAS EN:", SAVE_DIR)


if __name__ == "__main__":
    main()
