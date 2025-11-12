
import random, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_curves(history: dict, out_path, *, plot_val: bool = False):
    """
    Plot training curves. By default, ONLY train curves are drawn.
    Set plot_val=True to overlay validation if keys exist.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Loss ----
    plt.figure(figsize=(7, 5))
    plt.plot(history.get("train_loss", []), label="Train Loss")
    if plot_val and "val_loss" in history:
        plt.plot(history["val_loss"], label="Val Loss", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend() if plot_val else None
    plt.tight_layout()
    plt.savefig(out_path.with_name("loss_curve.png"), dpi=150)
    plt.close()

    # ---- Accuracy ----
    plt.figure(figsize=(7, 5))
    plt.plot(history.get("train_acc", []), label="Train Acc")
    if plot_val and "val_acc" in history:
        plt.plot(history["val_acc"], label="Val Acc", linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy")
    plt.legend() if plot_val else None
    plt.tight_layout()
    plt.savefig(out_path.with_name("acc_curve.png"), dpi=150)
    plt.close()

def plot_confusion_matrix_png(cm: np.ndarray, classes: list, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_text(text: str, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
