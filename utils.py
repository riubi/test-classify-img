import json
import os
import random

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, f1c, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "per_class": {
            c: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(f1c[i]), "support": int(sup[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_history(hist, best_epoch, title, path):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, key, name in zip(ax, ["loss", "acc", "f1"], ["Loss", "Accuracy", "Macro F1"]):
        a.plot(ep, hist["train_" + key], label="train")
        a.plot(ep, hist["val_" + key], label="val")
        a.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best epoch {best_epoch}")
        a.set_xlabel("epoch")
        a.set_title(name)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, classes, title, path):
    cm = np.array(cm)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(classes)
    size = max(6, 0.55 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("предсказанный класс")
    ax.set_ylabel("истинный класс")
    ax.set_title(title + "\n(цвет - доля по строке, число - количество)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
