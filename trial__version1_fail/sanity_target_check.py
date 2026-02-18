import numpy as np
import pandas as pd

FIX = r"data\vsgui10k_fixations.csv"
TRIAL = ["pid", "img_name", "tgt_id", "cue", "absent"]

K_LIST = [1, 3, 5]  # last K fixations

# filter params (match your preprocess)
MIN_DUR = 0.08
ALLOW_MIN = -0.10
ALLOW_MAX = 1.10

def clip01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

def point_in_tgt(x, y, tx, ty, tw, th):
    return (tx <= x <= tx + tw) and (ty <= y <= ty + th)

def mean_dist_to_tgt_center(xs, ys, tx, ty, tw, th):
    cx = tx + tw / 2.0
    cy = ty + th / 2.0
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(np.mean(d))

def main():
    df = pd.read_csv(FIX)

    # keep only valid gaze rows
    df = df[(df.FPOGV == 1) & (df.BPOGV == 1)]
    df = df[df.FPOGD >= MIN_DUR]
    df = df[(df.FPOGX >= ALLOW_MIN) & (df.FPOGX <= ALLOW_MAX) &
            (df.FPOGY >= ALLOW_MIN) & (df.FPOGY <= ALLOW_MAX)]

    df = df.sort_values(TRIAL + ["TIME"])

    # containers
    hit_counts = {k: 0 for k in K_LIST}
    trial_counts = {k: 0 for k in K_LIST}
    dist_list = {k: [] for k in K_LIST}

    dropped_no_target = 0
    dropped_too_short = 0

    for _, g in df.groupby(TRIAL, sort=False):
        if len(g) < max(K_LIST):
            dropped_too_short += 1
            continue

        r0 = g.iloc[0]
        tx, ty, tw, th = r0.get("tgt_x"), r0.get("tgt_y"), r0.get("tgt_width"), r0.get("tgt_height")
        if not np.isfinite(tx) or not np.isfinite(ty) or not np.isfinite(tw) or not np.isfinite(th):
            dropped_no_target += 1
            continue

        # Use RAW but clip to [0,1] for fair target geometry comparison
        xs_all = g["FPOGX"].astype(float).map(clip01).to_numpy()
        ys_all = g["FPOGY"].astype(float).map(clip01).to_numpy()

        for k in K_LIST:
            xs = xs_all[-k:]
            ys = ys_all[-k:]

            trial_counts[k] += 1
            # hit if ANY of last-k points is inside target bbox
            hit = any(point_in_tgt(x, y, tx, ty, tw, th) for x, y in zip(xs, ys))
            if hit:
                hit_counts[k] += 1

            dist_list[k].append(mean_dist_to_tgt_center(xs, ys, tx, ty, tw, th))

    print("Trials evaluated (per K):", trial_counts)
    print("Dropped too short:", dropped_too_short)
    print("Dropped no target:", dropped_no_target)
    print()

    for k in K_LIST:
        n = trial_counts[k]
        if n == 0:
            print(f"last{k}: no trials")
            continue
        hit_rate = hit_counts[k] / n
        d = np.array(dist_list[k], dtype=float)
        print(
            f"last{k}: hit_rate={hit_rate:.4f}  "
            f"mean_dist={d.mean():.4f}  median={np.median(d):.4f}  p90={np.quantile(d,0.9):.4f}"
        )

if __name__ == "__main__":
    main()
