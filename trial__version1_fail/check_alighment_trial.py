import os, json, gzip
import numpy as np

JSONL = "trials_with_elements.jsonl.gz"
SEG_ROOT = r"data\segmentation"

PID = "015f21"
IMG = "0ff0be.png"
MEDIA_ID = 63

def point_in_box(x, y, b):
    return (b["x1"] <= x <= b["x2"]) and (b["y1"] <= y <= b["y2"])

def load_seg(seg_path):
    with open(seg_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    H, W, _ = d["img"]["shape"]
    boxes = []
    for c in d.get("compos", []):
        try:
            x1 = int(c["column_min"]); x2 = int(c["column_max"])
            y1 = int(c["row_min"]);    y2 = int(c["row_max"])
        except:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2, "area": (x2-x1)*(y2-y1)})
    return W, H, boxes

def hit_ratio(points, boxes):
    hit = 0
    for x,y in points:
        inside = False
        for b in boxes:
            if point_in_box(x,y,b):
                inside = True
                break
        hit += int(inside)
    return hit / max(1, len(points))

def main():
    trial = None
    with gzip.open(JSONL, "rt", encoding="utf-8") as f:
        for line in f:
            tr = json.loads(line)
            if tr["key"]["pid"] == PID and tr["key"]["img_name"] == IMG and tr["meta"]["media_id"] == MEDIA_ID:
                trial = tr
                break
    if trial is None:
        print("Trial not found.")
        return

    seg_path = trial["segmentation"]["seg_path"]
    W, H, boxes = load_seg(seg_path)

    pts = [(fx["xy_seg"]["x"], fx["xy_seg"]["y"]) for fx in trial["scanpath"]]
    pts = np.array(pts, dtype=float)

    # candidate transforms
    def T_identity(p): return p
    def T_flip_y(p):  return np.column_stack([p[:,0], H - p[:,1]])
    def T_flip_x(p):  return np.column_stack([W - p[:,0], p[:,1]])
    def T_swap(p):    return np.column_stack([p[:,1] * (W/H), p[:,0] * (H/W)])  # rough swap scaling
    def T_swap_flip_y(p):
        q = T_swap(p)
        return np.column_stack([q[:,0], H - q[:,1]])

    transforms = {
        "identity": T_identity,
        "flip_y": T_flip_y,
        "flip_x": T_flip_x,
        "swap": T_swap,
        "swap_flip_y": T_swap_flip_y,
    }

    results = []
    for name, fn in transforms.items():
        q = fn(pts)
        r = hit_ratio(q.tolist(), boxes)
        results.append((name, r))

    results.sort(key=lambda x: x[1], reverse=True)
    print("Alignment test (higher non-bg hit ratio is better):")
    for name, r in results:
        print(f"{name:12s}  hit_ratio={r:.3f}")

    print("\nIf a non-identity transform wins by a large margin, that suggests misalignment.")

if __name__ == "__main__":
    main()
