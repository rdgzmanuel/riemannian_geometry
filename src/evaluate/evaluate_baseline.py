from __future__ import annotations

import argparse
import os
import matplotlib.pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from JSON"
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="experiments/checkpoints/baseline/mlp_metrics.json",
        help="Ruta al JSON con métricas de entrenamiento",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="plots",
        help="Directorio donde guardar los gráficos",
    )
    return parser.parse_args()


def load_metrics_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"No se encontró el archivo de métricas: {path}")


def plot_metrics(metrics: dict, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = metrics.get("epochs", list(range(1, len(metrics.get("train_loss", [])) + 1)))

    # Loss plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, metrics.get("train_loss", []), label="Train Loss")
    plt.plot(epochs, metrics.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_plot_baseline.png"))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, metrics.get("train_acc", []), label="Train Acc")
    plt.plot(epochs, metrics.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_plot_baseline.png"))
    plt.close()

    print(f"Gráficos guardados en '{save_dir}'")


def main():
    args = parse_args()
    metrics = load_metrics_json(args.metrics_json)
    plot_metrics(metrics, args.save_dir)


if __name__ == "__main__":
    main()
