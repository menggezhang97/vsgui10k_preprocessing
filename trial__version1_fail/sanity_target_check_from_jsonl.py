import json
import numpy as np

JSONL = "trials_with_elements.jsonl"
K_LIST = [1, 3, 5]

def point_in_box(x, y, box):
    return (box["x1"] <= x <= box["x2"]) and (box["y1"] <= y <= box["y2"])

def mean_dist_to_center(xs, ys, box):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(np.mean(d))

def main():
    hit_counts = {k: 0 for k in K_LIST}
    trial_counts = {k: 0 for k in K_LIST}
    dist_list = {k: [] for k in K_LIST}

    dropped_no_target = 0
    dropped_too_short = 0

    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            scan = t.get("scanpath", [])
            if len(scan) < max(K_LIST):
                dropped_too_short += 1
                continue

            box = t.get("target_seg_px", None)
            if box is None:
                dropped_no_target += 1
                continue

            xs_all = np.array([s["x_px"] for s in scan], dtype=float)
            ys_all = np.array([s["y_px"] for s in scan], dtype=float)

            for k in K_LIST:
                xs = xs_all[-k:]
                ys = ys_all[-k:]

                trial_counts[k] += 1
                hit = any(point_in_box(x, y, box) for x, y in zip(xs, ys))
                if hit:
                    hit_counts[k] += 1

                dist_list[k].append(mean_dist_to_center(xs, ys, box))

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
            f"mean_dist_px={d.mean():.2f}  median={np.median(d):.2f}  p90={np.quantile(d,0.9):.2f}"
        )

if __name__ == "__main__":
    main()
