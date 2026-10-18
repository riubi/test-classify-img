import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem
from sklearn.model_selection import train_test_split

OFFICIAL = "project-oceania/whoi-plankton"
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--min_n", type=int, default=101)
parser.add_argument("--max_n", type=int, default=200)
parser.add_argument("--source", default="nf-whoi/whoi-plankton")
parser.add_argument("--data_dir", default="data")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

fs = HfFileSystem()
src = f"datasets/{args.source}"
os.makedirs(args.data_dir, exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("splits", exist_ok=True)


def read_card(repo):
    with fs.open(f"datasets/{repo}/README.md", "r") as f:
        return f.read()


def label_names(card):
    meta = yaml.safe_load(card.split("---")[1])
    feats = meta["dataset_info"]["features"]
    names = [x for x in feats if x["name"] == "label"][0]["dtype"]["class_label"]["names"]
    return [names[k] for k in sorted(names, key=int)]


def open_parquet(path):
    return pq.ParquetFile(fs.open(path, "rb", cache_type="none"))


def retry(fn, *a, tries=10):
    for t in range(tries):
        try:
            return fn(*a)
        except Exception as e:
            if t == tries - 1:
                raise
            print("  retry:", type(e).__name__, str(e)[:80], flush=True)
            time.sleep(20 * (t + 1))


card = read_card(OFFICIAL)
rows = re.findall(r"^(\d+): (\S+)\s+[▇ ]*?([\d.]+)\s*$", card, re.M)
counts = pd.DataFrame([(int(i), n, int(float(c))) for i, n, c in rows], columns=["label", "class", "count"])
total = counts["count"].sum()
print("классов:", len(counts), "изображений:", total)
assert len(counts) == len(label_names(card)) and total == 3563596, "гистограмма в карточке не разобралась"

counts["selected"] = counts["count"].between(args.min_n, args.max_n)
counts.sort_values("count", ascending=False).to_csv("results/class_counts.csv", index=False)
sel = counts[counts["selected"]].sort_values("class").reset_index(drop=True)
print(f"классы с {args.min_n}..{args.max_n} изображениями: {len(sel)}")
print(sel[["class", "count"]].to_string(index=False))

src_names = label_names(read_card(args.source))
want = {src_names.index(c): c for c in sel["class"]}
files = sorted(fs.glob(f"{src}/data/*.parquet"))
print("источник:", args.source, "parquet-файлов:", len(files))


def index_file(path):
    pf = retry(open_parquet, path)
    found = []
    for rg in range(pf.metadata.num_row_groups):
        labels = retry(lambda: pf.read_row_group(rg, columns=["label"]).column("label").to_pylist())
        found += [(os.path.basename(path), rg, pos, want[lab]) for pos, lab in enumerate(labels) if lab in want]
    print("  проиндексирован", os.path.basename(path), flush=True)
    return found


index_path = os.path.join(args.data_dir, f"index_{args.source.replace('/', '_')}.csv")
if os.path.exists(index_path):
    need = pd.read_csv(index_path)
else:
    with ThreadPoolExecutor(args.workers) as ex:
        found = sum(ex.map(index_file, files), [])
    need = pd.DataFrame(found, columns=["file", "row_group", "pos", "class"])
    need.to_csv(index_path, index=False)

got = need["class"].value_counts().reindex(sel["class"]).fillna(0).astype(int)
print(pd.DataFrame({"official": sel["count"].values, "source": got.values}, index=sel["class"]))
assert (got.values == sel["count"].values).all(), "в источнике другое число картинок, чем в официальном датасете"

img_dir = os.path.join(args.data_dir, "images")
for cls in sel["class"]:
    os.makedirs(os.path.join(img_dir, cls), exist_ok=True)


def dump_file(item):
    fname, part = item
    pf = retry(open_parquet, f"{src}/data/{fname}")
    saved = []
    for rg, grp in part.groupby("row_group"):
        images = retry(lambda: pf.read_row_group(int(rg), columns=["image"]).column("image").to_pylist())
        for pos, cls in zip(grp["pos"], grp["class"]):
            im = images[pos]
            out = os.path.join(img_dir, cls, os.path.basename(im["path"]))
            if not os.path.exists(out):
                with open(out, "wb") as f:
                    f.write(im["bytes"])
            saved.append((os.path.relpath(out, args.data_dir), cls))
    return saved


groups = list(need.groupby("file"))
print("нужно прочитать row groups:", need.groupby(["file", "row_group"]).ngroups)
with ThreadPoolExecutor(args.workers) as ex:
    df = pd.DataFrame(sum(ex.map(dump_file, groups), []), columns=["path", "class"])
df = df.sort_values("path").reset_index(drop=True)
assert df["path"].is_unique, "одинаковые имена файлов"
assert (df["class"].value_counts().reindex(sel["class"]).values == sel["count"].values).all()

train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["class"], random_state=SEED)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["class"], random_state=SEED)
split = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")])
split = split.sort_values(["split", "class", "path"])
split.to_csv("splits/split.csv", index=False)

table = split.groupby(["class", "split"]).size().unstack()[["train", "val", "test"]]
table["total"] = table.sum(axis=1)
table.loc["ВСЕГО"] = table.sum()
table.to_csv("results/split_sizes.csv")
print(table)
