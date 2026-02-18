import json
import numpy as np

JSONL = "trials_with_elements.jsonl"

K_LIST = [1, 3, 5]
DILATE_R = [1.0, 1.5, 2.0, 3.0]  # bbox expansion ratio

def expand_box(box, r):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    w = (box["x2"] - box["x1"]) * r
    h = (box["y2"] - box["y1"]) * r
    return {"x1": cx - w/2, "y1": cy - h/2, "x2": cx + w/2, "y2": cy + h/2}

def point_in_box(x, y, box):
    return (box["x1"] <= x <= box["x2"]) and (box["y1"] <= y <= box["y2"])

def dist_to_center(x, y, box):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return float(np.sqrt((x-cx)**2 + (y-cy)**2))

def mean_dist(xs, ys, box):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(np.mean(d))

def main():
    # 1) near-target hit-rate with dilation
    hit = {(k,r):0 for k in K_LIST for r in DILATE_R}
    ntr = {(k,r):0 for k in K_LIST for r in DILATE_R}

    # 2) early/mid/late distance
    dist_early, dist_mid, dist_late = [], [], []

    dropped_no_target = 0
    dropped_short = 0

    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            scan = t.get("scanpath", [])
            box = t.get("target_seg_px", None)

            if box is None:
                dropped_no_target += 1
                continue

            if len(scan) < max(K_LIST):
                dropped_short += 1
                continue

            xs_all = np.array([s["x_px"] for s in scan], dtype=float)
            ys_all = np.array([s["y_px"] for s in scan], dtype=float)

            # (A) dilation hit rates
            for r in DILATE_R:
                box_r = expand_box(box, r)
                for k in K_LIST:
                    xs = xs_all[-k:]
                    ys = ys_all[-k:]
                    ntr[(k,r)] += 1
                    ok = any(point_in_box(x,y,box_r) for x,y in zip(xs,ys))
                    if ok:
                        hit[(k,r)] += 1

            # (B) early/mid/late distance
            n = len(xs_all)
            a = n // 3
            b = 2 * n // 3
            if a < 1 or (b-a) < 1 or (n-b) < 1:
                continue
            dist_early.append(mean_dist(xs_all[:a], ys_all[:a], box))
            dist_mid.append(mean_dist(xs_all[a:b], ys_all[a:b], box))
            dist_late.append(mean_dist(xs_all[b:], ys_all[b:], box))

    print("Dropped short:", dropped_short, "Dropped no target:", dropped_no_target)
    print()

    print("Near-target hit rate (bbox dilation):")
    for r in DILATE_R:
        row = []
        for k in K_LIST:
            rate = hit[(k,r)] / max(1, ntr[(k,r)])
            row.append(f"last{k}={rate:.3f}")
        print(f"  r={r}: " + "  ".join(row))

    print("\nDistance trend (px) early/mid/late:")
    e = np.array(dist_early); m = np.array(dist_mid); l = np.array(dist_late)
    print("  n =", len(e))
    print("  mean:", round(e.mean(),2), round(m.mean(),2), round(l.mean(),2))
    print("  median:", round(np.median(e),2), round(np.median(m),2), round(np.median(l),2))
    print("  p90:", round(np.quantile(e,0.9),2), round(np.quantile(m,0.9),2), round(np.quantile(l,0.9),2))

if __name__ == "__main__":
    main()
