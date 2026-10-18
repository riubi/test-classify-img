import argparse
import copy
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data import (IMAGENET_MEAN, IMAGENET_STD, PlanktonDataset, dataset_stats, get_transforms,
                  load_split, seed_worker)
from models import SimpleCNN, build_resnet18, count_params, freeze_bn_stats
from utils import compute_metrics, get_device, plot_confusion, plot_history, save_json, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
parser.add_argument("--mode", choices=["fc", "layer4", "layer3_4", "full"], default="fc")
parser.add_argument("--aug", choices=["none", "basic", "flip", "strong"], default="basic")
parser.add_argument("--act", default="relu")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--channels", default="32,64,128,256")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--opt", choices=["adam", "adamw", "sgd"], default="adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_backbone", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data_dir", default="data")
parser.add_argument("--out", default="results/runs")
args = parser.parse_args()

set_seed(args.seed)
device = get_device()
out_dir = os.path.join(args.out, args.name)
os.makedirs(out_dir, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = os.path.join("checkpoints", args.name + ".pt")

df, classes, class_to_idx = load_split()
n_classes = len(classes)
train_ds = PlanktonDataset(df[df.split == "train"], args.data_dir, class_to_idx)
val_ds = PlanktonDataset(df[df.split == "val"], args.data_dir, class_to_idx)
test_ds = PlanktonDataset(df[df.split == "test"], args.data_dir, class_to_idx)

if args.model == "resnet18":
    mean, std = IMAGENET_MEAN, IMAGENET_STD
else:
    m, s = dataset_stats(train_ds)
    mean, std = [m] * 3, [s] * 3

train_tf, eval_tf = get_transforms(args.aug, mean, std)
train_ds.transform = train_tf
val_ds.transform = eval_tf
test_ds.transform = eval_tf
train_eval_ds = copy.copy(train_ds)
train_eval_ds.transform = eval_tf

g = torch.Generator()
g.manual_seed(args.seed)
train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, generator=g, worker_init_fn=seed_worker)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)
train_eval_loader = DataLoader(train_eval_ds, batch_size=64)
print(f"classes: {n_classes}, train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}, device {device}")

if args.model == "cnn":
    channels = tuple(int(c) for c in args.channels.split(","))
    model = SimpleCNN(n_classes, act=args.act, dropout=args.dropout, channels=channels)
    param_groups = [{"params": model.parameters(), "lr": args.lr}]
else:
    model = build_resnet18(n_classes, args.mode)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    param_groups = [{"params": head, "lr": args.lr}]
    if body:
        param_groups.append({"params": body, "lr": args.lr_backbone})
model = model.to(device)
total_params, trainable_params = count_params(model)
print(f"params: total {total_params:,}, trainable {trainable_params:,}")

if args.opt == "adam":
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.wd)
elif args.opt == "adamw":
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)
else:
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss()


def run_epoch(loader, train=False):
    model.train(train)
    if train:
        freeze_bn_stats(model)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(out.argmax(1).cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    acc = float((preds == targets).mean())
    f1 = float(f1_score(targets, preds, average="macro", labels=list(range(n_classes)), zero_division=0))
    return total_loss / len(targets), acc, f1, preds, targets


hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1", "lr"]}
best_f1, best_loss, best_epoch, wait = -1.0, float("inf"), 0, 0
t_start = time.time()
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_f1, _, _ = run_epoch(val_loader)
    lr_now = optimizer.param_groups[-1]["lr"]
    for k, v in zip(hist, [tr_loss, va_loss, tr_acc, va_acc, tr_f1, va_f1, lr_now]):
        hist[k].append(v)
    scheduler.step()

    improved = va_f1 > best_f1 or (va_f1 == best_f1 and va_loss < best_loss)
    if improved:
        best_f1, best_loss, best_epoch, wait = va_f1, va_loss, epoch, 0
        torch.save(model.state_dict(), ckpt_path)
    else:
        wait += 1
    print(f"ep {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
          f"val loss {va_loss:.3f} acc {va_acc:.3f} f1 {va_f1:.3f} | lr {lr_now:.1e} | "
          f"{time.time() - t0:.0f}s {'*' if improved else ''}", flush=True)
    if wait >= args.patience:
        print(f"early stopping, лучшая эпоха {best_epoch}")
        break
train_time = time.time() - t_start

model.load_state_dict(torch.load(ckpt_path, map_location=device))
results = {"best_epoch": best_epoch, "epochs_trained": len(hist["train_loss"]), "train_time_sec": train_time}
for split_name, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
    loss, _, _, preds, targets = run_epoch(loader)
    results[split_name] = compute_metrics(targets, preds, classes)
    results[split_name]["loss"] = loss

config = vars(args).copy()
config.update({"classes": classes, "norm_mean": mean, "norm_std": std, "device": str(device),
               "total_params": total_params, "trainable_params": trainable_params})
save_json(config, os.path.join(out_dir, "config.json"))
save_json(hist, os.path.join(out_dir, "history.json"))
save_json(results, os.path.join(out_dir, "metrics.json"))
plot_history(hist, best_epoch, args.name, os.path.join(out_dir, "curves.png"))
plot_confusion(results["test"]["confusion_matrix"], classes, f"{args.name}: test",
               os.path.join(out_dir, "cm_test.png"))

t = results["test"]
print(f"TEST {args.name}: acc {t['accuracy']:.4f} bal_acc {t['balanced_accuracy']:.4f} "
      f"P {t['precision_macro']:.4f} R {t['recall_macro']:.4f} macro F1 {t['f1_macro']:.4f} "
      f"(best epoch {best_epoch}, {train_time / 60:.1f} min)")
