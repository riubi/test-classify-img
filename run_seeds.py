import json
import os
import subprocess
import sys

SEEDS = [1, 2]
KEYS = ["model", "mode", "aug", "act", "dropout", "channels", "bs", "opt", "lr", "lr_backbone", "wd", "epochs",
        "patience"]

with open("results/selection.json") as f:
    sel = json.load(f)
runs = list(dict.fromkeys([sel["base_cnn"], sel["best_cnn"], "resnet_fc", "resnet_layer4", "resnet_layer3_4",
                           "resnet_full"]))
os.makedirs("logs", exist_ok=True)

for run in runs:
    with open(f"results/runs/{run}/config.json") as f:
        cfg = json.load(f)
    for seed in SEEDS:
        name = f"{run}_s{seed}"
        if os.path.exists(f"results/seeds/{name}/metrics.json"):
            continue
        cmd = [sys.executable, "-u", "train.py", "--name", name, "--seed", str(seed), "--out", "results/seeds"]
        for k in KEYS:
            if k in cfg:
                cmd += [f"--{k}", str(cfg[k])]
        with open(f"logs/{name}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        with open(f"logs/{name}.log") as log:
            print([line.strip() for line in log if line.startswith("TEST")][-1], flush=True)
