import glob
import json
import os
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_transforms, load_image, load_split

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)

LAYERS = {"fc": "FC", "layer4": "layer4 + FC", "layer3_4": "layer3-4 + FC", "full": "Full fine-tuning"}


def load_runs():
    runs = {}
    for d in sorted(glob.glob("results/runs/*")):
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        name = os.path.basename(d)
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(d, "metrics.json")) as f:
            met = json.load(f)
        with open(os.path.join(d, "history.json")) as f:
            hist = json.load(f)
        runs[name] = {"cfg": cfg, "met": met, "hist": hist}
    return runs


def lr_str(cfg):
    if cfg["model"] == "resnet18" and cfg["mode"] != "fc":
        return f"{cfg['lr']:g} / {cfg['lr_backbone']:g}"
    return f"{cfg['lr']:g}"


def summary_table(runs):
    rows = []
    for name, r in runs.items():
        c, m = r["cfg"], r["met"]
        t = m["test"]
        rows.append({
            "run": name,
            "model": "Custom CNN" if c["model"] == "cnn" else "ResNet18",
            "trainable_layers": "all" if c["model"] == "cnn" else LAYERS[c["mode"]],
            "trainable_params": c["trainable_params"],
            "aug": c["aug"],
            "batch_size": c["bs"],
            "activation": c["act"] if c["model"] == "cnn" else "relu",
            "dropout": c["dropout"] if c["model"] == "cnn" else None,
            "channels": c.get("channels", "32,64,128,256") if c["model"] == "cnn" else None,
            "optimizer": c["opt"],
            "lr": lr_str(c),
            "wd": c["wd"],
            "best_epoch": m["best_epoch"],
            "epochs": m["epochs_trained"],
            "train_f1": m["train"]["f1_macro"],
            "val_f1": m["val"]["f1_macro"],
            "test_accuracy": t["accuracy"],
            "test_balanced_acc": t["balanced_accuracy"],
            "test_precision": t["precision_macro"],
            "test_recall": t["recall_macro"],
            "test_f1": t["f1_macro"],
            "train_loss": m["train"]["loss"],
            "val_loss": m["val"]["loss"],
            "test_loss": t["loss"],
            "time_min": m["train_time_sec"] / 60,
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False, float_format="%.4f")
    return df


def plot_curves(runs, names, labels, path, title):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, lab in zip(names, labels):
        if name not in runs:
            continue
        h = runs[name]["hist"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        line, = ax[0].plot(ep, h["train_loss"], ls="--", alpha=0.8)
        ax[0].plot(ep, h["val_loss"], color=line.get_color(), label=lab)
        ax[1].plot(ep, h["train_acc"], ls="--", color=line.get_color(), alpha=0.8)
        ax[1].plot(ep, h["val_acc"], color=line.get_color(), label=lab)
        ax[2].plot(ep, h["val_f1"], color=line.get_color(), label=lab)
    ax[0].set_title("Loss (пунктир - train, сплошная - val)")
    ax[1].set_title("Accuracy (пунктир - train, сплошная - val)")
    ax[2].set_title("Val macro F1")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bars(df, names, labels, path, title):
    d = df.set_index("run").reindex([n for n in names if n in set(df.run)])
    labels = [l for n, l in zip(names, labels) if n in set(df.run)]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(d) + 2), 4.2))
    ax.bar(x - 0.2, d["val_f1"], 0.4, label="val macro F1")
    ax.bar(x + 0.2, d["test_f1"], 0.4, label="test macro F1")
    for i, (v, t) in enumerate(zip(d["val_f1"], d["test_f1"])):
        ax.text(i - 0.2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, t + 0.005, f"{t:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    lo = max(0, min(d["val_f1"].min(), d["test_f1"].min()) - 0.1)
    ax.set_ylim(lo, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_class(runs, names, labels, path):
    names = [n for n in names if n in runs]
    classes = runs[names[0]]["cfg"]["classes"]
    mat = np.array([[runs[n]["met"]["test"]["per_class"][c]["f1"] for n in names] for c in classes])
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 2.8, 0.42 * len(classes) + 1.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_title("F1 по классам на test")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return pd.DataFrame(mat, index=classes, columns=names)


def plot_examples(path, per_class=6):
    df, classes, _ = load_split()
    rng = random.Random(0)
    fig, ax = plt.subplots(len(classes), per_class, figsize=(per_class * 1.4, len(classes) * 1.45))
    for i, c in enumerate(classes):
        paths = sorted(df[df["class"] == c]["path"])
        for j, p in enumerate(rng.sample(paths, per_class)):
            ax[i, j].imshow(load_image(os.path.join("data", p)), cmap="gray", vmin=0, vmax=255)
            ax[i, j].axis("off")
        ax[i, 0].set_title(c, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_aug_examples(path, n=7):
    df, classes, _ = load_split()
    rng = random.Random(1)
    picks = [rng.choice(sorted(df[df["class"] == c]["path"])) for c in rng.sample(classes, 3)]
    random.seed(0)
    import torch
    torch.manual_seed(0)
    fig, ax = plt.subplots(len(picks) * 2, n + 1, figsize=((n + 1) * 1.4, len(picks) * 2 * 1.5))
    for k, p in enumerate(picks):
        img = load_image(os.path.join("data", p))
        for r, aug in enumerate(["basic", "strong"]):
            tf, _ = get_transforms(aug, [0.0] * 3, [1.0] * 3)
            row = 2 * k + r
            ax[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
            ax[row, 0].set_title("исходное" if r == 0 else "", fontsize=8)
            for j in range(1, n + 1):
                x = tf(img)[0].numpy()
                ax[row, j].imshow(x, cmap="gray", vmin=0, vmax=1)
            ax[row, 1].set_title(aug, fontsize=8, loc="left")
            for a in ax[row]:
                a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_class_counts(path):
    counts = pd.read_csv("results/class_counts.csv").sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    colors = ["tab:orange" if s else "tab:blue" for s in counts["selected"]]
    ax.bar(range(len(counts)), counts["count"].clip(lower=1), color=colors)
    ax.set_yscale("log")
    ax.axhline(101, color="gray", ls="--", lw=1)
    ax.axhline(200, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts["class"], rotation=90, fontsize=6)
    ax.set_ylabel("изображений (log)")
    ax.set_title("Число изображений в классах WHOI-Plankton (оранжевые - выбранные, 101..200)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_confusions(runs, names, k=5):
    rows = []
    for n in names:
        if n not in runs:
            continue
        classes = runs[n]["cfg"]["classes"]
        cm = np.array(runs[n]["met"]["test"]["confusion_matrix"])
        sup = cm.sum(axis=1)
        pairs = [(cm[i, j], i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
        for cnt, i, j in sorted(pairs, reverse=True)[:k]:
            rows.append({"run": n, "true": classes[i], "pred": classes[j], "count": int(cnt),
                         "share_of_true": cnt / sup[i]})
    return pd.DataFrame(rows)


def seeds_table(names, main_dir="results/runs", seed_dir="results/seeds"):
    rows = []
    for n in names:
        vals = []
        for d in [os.path.join(main_dir, n)] + sorted(glob.glob(os.path.join(seed_dir, n + "_s*"))):
            p = os.path.join(d, "metrics.json")
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                vals.append((m["test"]["accuracy"], m["test"]["f1_macro"], m["val"]["f1_macro"]))
        if len(vals) > 1:
            v = np.array(vals)
            rows.append({"run": n, "n_seeds": len(v),
                         "test_acc_mean": v[:, 0].mean(), "test_acc_std": v[:, 0].std(ddof=1),
                         "test_f1_mean": v[:, 1].mean(), "test_f1_std": v[:, 1].std(ddof=1),
                         "val_f1_mean": v[:, 2].mean(), "val_f1_std": v[:, 2].std(ddof=1)})
    return pd.DataFrame(rows)


def best_by_val(df, names):
    d = df.set_index("run").loc[[n for n in names if n in set(df.run)]]
    return d["val_f1"].idxmax()


if __name__ == "__main__":
    runs = load_runs()
    df = summary_table(runs)
    print(df[["run", "aug", "activation", "batch_size", "optimizer", "lr", "dropout", "best_epoch",
              "train_f1", "val_f1", "test_accuracy", "test_f1"]].to_string(index=False))

    plot_class_counts(os.path.join(FIG, "class_counts.png"))
    plot_examples(os.path.join(FIG, "examples.png"))
    plot_aug_examples(os.path.join(FIG, "aug_examples.png"))

    aug_runs = ["cnn_noaug", "cnn_aug", "cnn_aug_strong"]
    plot_curves(runs, aug_runs, ["без аугментаций", "basic", "strong"],
                os.path.join(FIG, "aug_curves.png"), "Своя CNN: влияние аугментаций")
    base = best_by_val(df, aug_runs)
    plot_curves(runs, ["cnn_noaug", "cnn_aug", "cnn_aug_flip"], ["без аугментаций", "basic", "flip"],
                os.path.join(FIG, "aug_flip_curves.png"), "Своя CNN: basic и flip (без поворотов на 90°)")

    cnn_runs = [base] + sorted(n for n in runs if n.startswith("cnn_") and n not in aug_runs + ["cnn_aug_flip"])
    plot_bars(df, cnn_runs, cnn_runs, os.path.join(FIG, "cnn_params.png"), "Своя CNN: эксперименты")
    for group, names in {
        "act": [base, "cnn_leaky_relu", "cnn_gelu"],
        "bs": ["cnn_bs16", base, "cnn_bs64"],
        "lr": ["cnn_lr3e-4", base, "cnn_lr3e-3"],
        "opt": [base, "cnn_adamw", "cnn_sgd"],
        "dropout": ["cnn_drop0.2", base, "cnn_drop0.7"],
        "depth": [base, "cnn_5blocks", "cnn_best"],
    }.items():
        plot_curves(runs, names, names, os.path.join(FIG, f"cnn_{group}_curves.png"), f"Своя CNN: {group}")

    res_runs = ["resnet_fc", "resnet_layer4", "resnet_layer3_4", "resnet_full"]
    res_labels = [LAYERS[n.split("_", 1)[1]] for n in res_runs]
    plot_curves(runs, res_runs, res_labels, os.path.join(FIG, "resnet_curves.png"), "ResNet18: режимы обучения")

    aug_res = [r for r in res_runs if r + "_aug" in runs]
    if aug_res:
        names = [n for r in aug_res for n in (r, r + "_aug")]
        labels = [f"{LAYERS[r.split('_', 1)[1]]}{', aug' if a else ''}" for r in aug_res for a in (0, 1)]
        plot_bars(df, names, labels, os.path.join(FIG, "resnet_aug.png"), "ResNet18 без аугментаций и с basic")

    best_cnn = best_by_val(df, cnn_runs)
    with open("results/selection.json", "w") as f:
        json.dump({"base_cnn": base, "best_cnn": best_cnn}, f, indent=2)
    print("base:", base, "best cnn:", best_cnn)
    cnn_final = [base] if best_cnn == base else [base, best_cnn]
    cnn_labels = ["Custom CNN"] if best_cnn == base else ["Custom CNN (база)", "Custom CNN (лучшая)"]
    final = cnn_final + res_runs
    final_labels = cnn_labels + ["ResNet18 " + lab for lab in res_labels]
    plot_bars(df, final, final_labels, os.path.join(FIG, "final_compare.png"), "Итоговое сравнение моделей")
    pc = plot_per_class(runs, final, final_labels, os.path.join(FIG, "per_class_f1.png"))
    pc.to_csv("results/per_class_f1.csv", float_format="%.3f")
    print(pc.round(3))
    conf = top_confusions(runs, final)
    conf.to_csv("results/top_confusions.csv", index=False, float_format="%.3f")
    print(conf.to_string(index=False))
    seeds = seeds_table(final)
    if len(seeds):
        seeds.to_csv("results/seeds_summary.csv", index=False, float_format="%.4f")
        print(seeds.round(3).to_string(index=False))
