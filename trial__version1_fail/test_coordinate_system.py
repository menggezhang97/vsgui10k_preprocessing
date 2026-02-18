import os
import glob
import json
import random
import numpy as np
import pandas as pd

random.seed(7)

FIX = r"data\vsgui10k_fixations.csv"
SEG_ROOT = r"data\segmentation"

TRIAL = ["pid", "img_name", "tgt_id", "cue", "absent"]

print("Loading fixations...")
df = pd.read_csv(FIX)
df = df[(df.FPOGV == 1) & (df.BPOGV == 1)]

print("Building segmentation index...")
seg_idx = {}
for p in glob.glob(os.path.join(SEG_ROOT, "block *", "*.json")):
    stem = os.path.splitext(os.path.basename(p))[0]
    seg_idx.setdefault(stem, p)

print("Sampling trials...")
trials = list(df.groupby(TRIAL).groups.keys())
random.shuffle(trials)
trials = trials[:300]

cols = [
    ("raw", "FPOGX", "FPOGY"),
    ("debias", "FPOGX_debias", "FPOGY_debias"),
    ("scaled", "FPOGX_scaled", "FPOGY_scaled"),
]

gb = df.groupby(TRIAL)
acc = {k: [] for k, _, _ in cols}

print("Running evaluation...")

for key in trials:
    g = gb.get_group(key)
    img = str(key[1])
    stem = os.path.splitext(img)[0]
    seg_path = seg_idx.get(stem)
    if not seg_path:
        continue

    d = json.load(open(seg_path, "r", encoding="utf-8"))
    H, W, _ = d["img"]["shape"]

    for name, xcol, ycol in cols:
        xy = g[[xcol, ycol]].dropna().to_numpy()
        if len(xy) == 0:
            continue

        in01 = ((xy[:, 0] >= 0) & (xy[:, 0] <= 1) &
                (xy[:, 1] >= 0) & (xy[:, 1] <= 1)).mean()

        x = xy[:, 0] * W
        y = xy[:, 1] * H

        inseg = ((x >= 0) & (x <= W) &
                 (y >= 0) & (y <= H)).mean()

        acc[name].append((in01, inseg))

print("\nResults:")
for name in acc:
    a = np.array(acc[name])
    if len(a) == 0:
        print(name, "no data")
        continue

    print(
        name,
        "n =", len(a),
        "mean_in01 =", round(a[:, 0].mean(), 3),
        "mean_inseg =", round(a[:, 1].mean(), 3)
    )
