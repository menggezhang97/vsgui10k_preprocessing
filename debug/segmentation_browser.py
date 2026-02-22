import os
import json
import glob
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional visualization (only used if --viz is enabled)
try:
    from PIL import Image
    import matplotlib.pyplot as plt
except Exception:
    Image = None
    plt = None


@dataclass
class SegFileSummary:
    path: str
    stem: str
    w: int
    h: int
    n_compos: int
    class_counts: Counter
    area_list: List[float]  # bbox areas in seg space
    has_bbox: int
    only_bg_image: int


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_json(p: str) -> Optional[dict]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_shape(d: dict) -> Tuple[int, int]:
    # Expected: d["img"]["shape"] = [H, W, 3]
    try:
        H, W, _ = d["img"]["shape"]
        return int(W), int(H)
    except Exception:
        return 0, 0


def _extract_compos(d: dict) -> List[dict]:
    # Your schema: d["compos"] is list of components
    comp = d.get("compos", [])
    if isinstance(comp, list):
        return comp
    return []


def _bbox_from_compo(c: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    VSGUI10K segmentation format (as you showed):
      column_min, column_max, row_min, row_max
    Return (x0,y0,x1,y1)
    """
    if not isinstance(c, dict):
        return None
    keys = ["column_min", "column_max", "row_min", "row_max"]
    if not all(k in c for k in keys):
        return None
    try:
        x0 = float(c["column_min"])
        x1 = float(c["column_max"])
        y0 = float(c["row_min"])
        y1 = float(c["row_max"])
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    except Exception:
        return None


def summarize_one(seg_path: str) -> Optional[SegFileSummary]:
    d = _load_json(seg_path)
    if d is None:
        return None

    w, h = _extract_shape(d)
    compos = _extract_compos(d)

    class_counts = Counter()
    area_list = []
    has_bbox = 0

    for c in compos:
        cls = c.get("class", "UNKNOWN")
        class_counts[str(cls)] += 1
        bb = _bbox_from_compo(c)
        if bb is not None:
            has_bbox += 1
            x0, y0, x1, y1 = bb
            area_list.append((x1 - x0) * (y1 - y0))

    # Check if it is "only Background + Image" style
    # (Some files may have only these coarse components)
    only_bg_image = 0
    unique_classes = set(class_counts.keys())
    if unique_classes.issubset({"Background", "Image"}) and len(unique_classes) <= 2:
        only_bg_image = 1

    stem = os.path.splitext(os.path.basename(seg_path))[0]
    return SegFileSummary(
        path=seg_path,
        stem=stem,
        w=w, h=h,
        n_compos=len(compos),
        class_counts=class_counts,
        area_list=area_list,
        has_bbox=has_bbox,
        only_bg_image=only_bg_image,
    )


def percentile_stats(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {}
    a = np.asarray(arr, dtype=float)
    return {
        "n": float(a.size),
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def find_seg_files(seg_root: str) -> List[str]:
    # Your structure: segmentation/block 0/*.json ... block 19/*.json
    pattern = os.path.join(seg_root, "block *", "*.json")
    files = glob.glob(pattern)
    if files:
        return sorted(files)
    # fallback recursive
    return sorted(glob.glob(os.path.join(seg_root, "**", "*.json"), recursive=True))


def draw_overlay_example(seg_summary: SegFileSummary, img_dir: str, out_dir: str,
                         max_boxes: int = 300, min_area: float = 16.0,
                         hide_bg_image: bool = True):
    """
    Optional visualization: draw UI component boxes on resized screenshot.
    Needs PIL + matplotlib.
    """
    if Image is None or plt is None:
        print("[WARN] PIL/matplotlib not available; skip visualization.")
        return

    # image name: stem + ".png" or ".jpg" ?
    # In your dataset, screenshots are in vsgui10k-images with the full filename.
    # Here we try common extensions.
    candidates = [
        os.path.join(img_dir, seg_summary.stem + ext)
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    ]
    img_path = None
    for c in candidates:
        if os.path.exists(c):
            img_path = c
            break

    if img_path is None:
        print(f"[VIZ] No matching image for stem={seg_summary.stem}. Skipped.")
        return

    d = _load_json(seg_summary.path)
    compos = _extract_compos(d)

    img = Image.open(img_path).convert("RGB")
    # Resize to seg space so bbox coordinates align
    img = img.resize((seg_summary.w, seg_summary.h), Image.BILINEAR)

    boxes = []
    for c in compos:
        cls = str(c.get("class", "UNKNOWN"))
        if hide_bg_image and cls in ("Background", "Image"):
            continue
        bb = _bbox_from_compo(c)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        area = (x1 - x0) * (y1 - y0)
        if area < min_area:
            continue
        boxes.append((x0, y0, x1, y1, cls))

    # cap
    boxes = boxes[:max_boxes]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"viz_{seg_summary.stem}.png")

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"{seg_summary.stem}  n_compos={seg_summary.n_compos}  classes={len(seg_summary.class_counts)}")

    for (x0, y0, x1, y1, cls) in boxes:
        plt.plot([x0, x1, x1, x0, x0],
                 [y0, y0, y1, y1, y0],
                 linewidth=1)
        # label optional (too many will clutter)
        # plt.text(x0+2, y0+10, cls, fontsize=6, bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"))

    plt.xlim([0, seg_summary.w])
    plt.ylim([seg_summary.h, 0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[VIZ] Saved: {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Browse VSGUI10K segmentation JSONs to assess UI mapping quality."
    )
    ap.add_argument("--data_root", type=str, default="data",
                    help="Path to dataset root containing segmentation/ and vsgui10k-images/. (default: data)")
    ap.add_argument("--n_sample", type=int, default=20,
                    help="Number of random files to print in detail. (default: 20)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--viz", action="store_true",
                    help="Also generate a few overlay images with UI boxes (requires PIL+matplotlib).")
    ap.add_argument("--viz_n", type=int, default=5,
                    help="How many overlay images to generate if --viz. (default: 5)")
    args = ap.parse_args()

    random.seed(args.seed)

    seg_root = os.path.join(args.data_root, "segmentation")
    img_dir = os.path.join(args.data_root, "vsgui10k-images")

    files = find_seg_files(seg_root)
    if not files:
        print(f"[ERROR] No segmentation json found under: {seg_root}")
        return

    print(f"Found {len(files)} segmentation files under: {seg_root}")

    summaries = []
    for p in files:
        s = summarize_one(p)
        if s is not None:
            summaries.append(s)

    print(f"Parsed {len(summaries)} / {len(files)} files.")

    n_compos_list = [s.n_compos for s in summaries]
    only_bg_image_count = sum(s.only_bg_image for s in summaries)
    has_bbox_rate = float(sum(1 for s in summaries if s.has_bbox > 0)) / float(len(summaries))

    # Aggregate class counts
    cls_all = Counter()
    for s in summaries:
        cls_all.update(s.class_counts)

    # Aggregate area stats across all boxes
    all_areas = []
    for s in summaries:
        all_areas.extend(s.area_list)

    print("\n=== Global Stats ===")
    print(f"Files: {len(summaries)}")
    print(f"Has any bbox: {has_bbox_rate:.3f}")
    print(f"Only Background/Image files: {only_bg_image_count} ({only_bg_image_count/len(summaries):.3f})")

    print("\nCompos count stats (每张图检测到多少组件):")
    print(percentile_stats([float(x) for x in n_compos_list]))

    print("\nBBox area stats in segmentation space (组件框面积分布):")
    print(percentile_stats([float(a) for a in all_areas]))

    print("\nTop classes (UI组件类别分布):")
    for cls, c in cls_all.most_common(20):
        print(f"  {cls:20s}  {c}")

    # Detailed samples
    sample_n = min(args.n_sample, len(summaries))
    sample = random.sample(summaries, sample_n)

    print("\n=== Random Samples (detailed) ===")
    for s in sample:
        top5 = s.class_counts.most_common(8)
        top5_str = ", ".join([f"{k}:{v}" for k, v in top5])
        print(f"\n- {s.stem}")
        print(f"  path: {s.path}")
        print(f"  shape (W,H): ({s.w},{s.h})")
        print(f"  n_compos: {s.n_compos}")
        print(f"  classes: {len(s.class_counts)}  [{top5_str}]")
        print(f"  only_bg_image: {s.only_bg_image}")
        print(f"  bbox_area_stats: {percentile_stats([float(a) for a in s.area_list])}")

    # Optional visualization
    if args.viz:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_viz_samples")
        viz_list = sorted(summaries, key=lambda x: x.n_compos, reverse=True)
        viz_list = viz_list[:max(args.viz_n, 1)]
        print(f"\n=== Visualization ({len(viz_list)} samples) ===")
        for s in viz_list:
            draw_overlay_example(s, img_dir=img_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()