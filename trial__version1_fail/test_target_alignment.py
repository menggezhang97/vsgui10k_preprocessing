import numpy as np
import pandas as pd

FIX = r"data\vsgui10k_fixations.csv"
TRIAL = ["pid", "img_name", "tgt_id", "cue", "absent"]
K = 5  # last K fixations

df = pd.read_csv(FIX)
df = df[(df.FPOGV == 1) & (df.BPOGV == 1)]
df = df.sort_values(TRIAL + ["TIME"])

cols = [
    ("raw", "FPOGX", "FPOGY"),
    ("debias", "FPOGX_debias", "FPOGY_debias"),
    ("scaled", "FPOGX_scaled", "FPOGY_scaled"),
]

def mean_lastk_dist(g, xcol, ycol):
    g2 = g[[xcol, ycol, "tgt_x", "tgt_y", "tgt_width", "tgt_height"]].dropna()
    if len(g2) < K:
        return None

    # target center in normalized coords
    tx = float(g2["tgt_x"].iloc[0]) + float(g2["tgt_width"].iloc[0]) / 2.0
    ty = float(g2["tgt_y"].iloc[0]) + float(g2["tgt_height"].iloc[0]) / 2.0

    # last K fixations
    tail = g2.tail(K)
    x = tail[xcol].astype(float).to_numpy()
    y = tail[ycol].astype(float).to_numpy()

    # Euclidean distance in normalized coordinate space
    d = np.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    return float(np.mean(d))

results = {name: [] for name, _, _ in cols}

for _, g in df.groupby(TRIAL, sort=False):
    for name, xcol, ycol in cols:
        d = mean_lastk_dist(g, xcol, ycol)
        if d is not None and np.isfinite(d):
            results[name].append(d)

print("Trials evaluated:", min(len(v) for v in results.values()))
for name in results:
    arr = np.array(results[name], dtype=float)
    print(
        name,
        "n=", len(arr),
        "mean_dist=", round(arr.mean(), 4),
        "median_dist=", round(np.median(arr), 4),
        "p90=", round(np.quantile(arr, 0.9), 4),
    )
