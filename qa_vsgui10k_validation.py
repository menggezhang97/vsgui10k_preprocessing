import os
import json
import glob
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# Config (run from: ...\VSGUI10K\data)
# =========================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

FIX_CSV = os.path.join(DATA_DIR, "vsgui10k_fixations.csv")
TGT_CSV = os.path.join(DATA_DIR, "vsgui10k_targets.csv")
AIM_CSV = os.path.join(DATA_DIR, "vsgui10k_aim_results.csv")  # optional, not required

IMG_DIR = os.path.join(DATA_DIR, "vsgui10k-images")
SEG_ROOT = os.path.join(DATA_DIR, "segmentation")

OUT_REPORT = os.path.join(BASE, "validation_report.json")
DEBUG_DIR = os.path.join(BASE, "validation_debug")

# Filtering
MIN_FIX_DUR = 0.08   # seconds
MIN_FIX_PER_TRIAL = 3

# Phases
IMG_TYPE_TGT_DESC = 0
IMG_TYPE_CROSS = 1
IMG_TYPE_SEARCH = 2
IMG_TYPE_VALID = 3

# Sampling
RANDOM_SEED = 7
SAMPLE_TRIALS_SEARCH_DEBUG = 20
SAMPLE_TRIALS_CROSS_DEBUG = 12
MAX_POINTS_DRAW = 80

# Target hit thresholds (bbox dilation ratio)
R_LIST = [1.0, 1.5, 2.0, 3.0]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================
# Helpers
# =========================
def ensure_dirs():
    os.makedirs(DEBUG_DIR, exist_ok=True)

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def build_seg_index(seg_root: str) -> Dict[str, str]:
    idx = {}
    pattern = os.path.join(seg_root, "block *", "*.json")
    for p in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(p))[0]
        idx.setdefault(stem, p)
    if not idx:
        for p in glob.glob(os.path.join(seg_root, "**", "*.json"), recursive=True):
            stem = os.path.splitext(os.path.basename(p))[0]
            idx.setdefault(stem, p)
    return idx

def load_seg_shape(seg_path: str) -> Tuple[int, int]:
    with open(seg_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    H, W, _ = d["img"]["shape"]
    return int(W), int(H)

def map_px_to_seg(x_px, y_px, orig_W, orig_H, W_seg, H_seg) -> Tuple[float, float]:
    rx = W_seg / float(orig_W)
    ry = H_seg / float(orig_H)
    return float(x_px * rx), float(y_px * ry)

def is_probably_normalized(v: float) -> bool:
    # robust-ish heuristic:
    # normalized gaze/bbox are usually in [0,1] (sometimes a bit out-of-range).
    return -2.0 <= v <= 2.0

def to_px_maybe(v: float, full: float) -> float:
    # If value looks normalized, convert to pixels; else assume already pixels.
    if is_probably_normalized(v):
        return float(v * full)
    return float(v)

def bbox_from_raw(r0, orig_W, orig_H) -> Optional[Tuple[float, float, float, float]]:
    # r0 has tgt_x/tgt_y/tgt_width/tgt_height
    tx = safe_float(r0.get("tgt_x"))
    ty = safe_float(r0.get("tgt_y"))
    tw = safe_float(r0.get("tgt_width"))
    th = safe_float(r0.get("tgt_height"))
    if None in (tx, ty, tw, th):
        return None
    x = to_px_maybe(tx, orig_W)
    y = to_px_maybe(ty, orig_H)
    w = to_px_maybe(tw, orig_W)
    h = to_px_maybe(th, orig_H)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)

def dilate_bbox_xyxy(b, r: float, W: float, H: float):
    x0, y0, x1, y1 = b
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = (x1 - x0) * r
    h = (y1 - y0) * r
    nx0 = clamp(cx - 0.5 * w, 0, W)
    nx1 = clamp(cx + 0.5 * w, 0, W)
    ny0 = clamp(cy - 0.5 * h, 0, H)
    ny1 = clamp(cy + 0.5 * h, 0, H)
    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return (nx0, ny0, nx1, ny1)

def point_in_bbox(x, y, b):
    x0, y0, x1, y1 = b
    return (x0 <= x <= x1) and (y0 <= y <= y1)

def trial_id_from_key(pid, img_name, tgt_id, cue, absent):
    return f"{pid}|{img_name}|{tgt_id}|{cue}|abs{int(bool(absent))}"

def render_debug_search(img_path, W_seg, H_seg, pts, tgt_box_seg, title, out_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_seg, H_seg), Image.BILINEAR)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)

    if tgt_box_seg is not None:
        x0, y0, x1, y1 = tgt_box_seg
        plt.plot([x0, x1, x1, x0, x0],
                 [y0, y0, y1, y1, y0],
                 linewidth=3, color="red")

    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, linewidth=2, color="cyan")
        plt.scatter(xs, ys, s=55, marker="o", c="yellow", edgecolors="black", linewidths=0.7)
        for i, (x, y) in enumerate(pts, start=1):
            plt.text(x + 4, y - 4, str(i),
                     fontsize=9, color="black",
                     bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

    plt.xlim([0, W_seg])
    plt.ylim([H_seg, 0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def render_debug_cross(img_path, W_seg, H_seg, pts, title, out_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_seg, H_seg), Image.BILINEAR)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)

    # Draw screen center crosshair
    cx = 0.5 * W_seg
    cy = 0.5 * H_seg
    plt.plot([cx - 12, cx + 12], [cy, cy], linewidth=2, color="lime")
    plt.plot([cx, cx], [cy - 12, cy + 12], linewidth=2, color="lime")

    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.scatter(xs, ys, s=55, marker="o", c="yellow", edgecolors="black", linewidths=0.7, alpha=0.85)

    plt.xlim([0, W_seg])
    plt.ylim([H_seg, 0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# Coordinate candidates
# =========================
@dataclass
class CoordCandidate:
    name: str
    x_col: str
    y_col: str
    # how to interpret values: "norm_or_px" -> use heuristic to_px_maybe;
    # if you KNOW it's px, set mode="px"
    mode: str = "norm_or_px"

CANDIDATES = [
    CoordCandidate("raw(FPOGX/FPOGY)", "FPOGX", "FPOGY", "norm_or_px"),
    CoordCandidate("debias(FPOGX_debias/FPOGY_debias)", "FPOGX_debias", "FPOGY_debias", "norm_or_px"),
    CoordCandidate("scaled(FPOGX_scaled/FPOGY_scaled)", "FPOGX_scaled", "FPOGY_scaled", "norm_or_px"),
]


# =========================
# Main validation
# =========================
def main():
    # Basic checks
    assert os.path.exists(FIX_CSV), f"Missing: {FIX_CSV}"
    assert os.path.exists(TGT_CSV), f"Missing: {TGT_CSV}"
    assert os.path.isdir(IMG_DIR), f"Missing: {IMG_DIR}"
    assert os.path.isdir(SEG_ROOT), f"Missing: {SEG_ROOT}"
    ensure_dirs()

    seg_idx = build_seg_index(SEG_ROOT)
    print("Seg index size:", len(seg_idx))

    fix = pd.read_csv(FIX_CSV, low_memory=False)
    tgt = pd.read_csv(TGT_CSV, low_memory=False)

    # -------------------------
    # (A) JOIN sanity: fix ↔ targets
    # Correct join: fix.img_name (UEyes namespace) ↔ targets.ueyes_img_name
    # -------------------------
    fix_pairs = fix[["img_name", "tgt_id"]].drop_duplicates()
    tgt_pairs = tgt[["ueyes_img_name", "tgt_id"]].drop_duplicates()

    joined = fix_pairs.merge(
        tgt_pairs,
        left_on=["img_name", "tgt_id"],
        right_on=["ueyes_img_name", "tgt_id"],
        how="inner"
    )

    join_report = {
        "fix_unique_pairs": int(len(fix_pairs)),
        "tgt_unique_pairs": int(len(tgt_pairs)),
        "joined_pairs": int(len(joined)),
        "join_rate_vs_fix_pairs": float(len(joined) / max(1, len(fix_pairs))),
        "note": "Correct join uses targets.ueyes_img_name (fixations img_name is UEyes namespace)."
    }

    # -------------------------
    # (B) Global filters (valid fixations)
    # -------------------------
    fix_valid = fix[(fix["FPOGV"] == 1) & (fix["BPOGV"] == 1)]
    fix_valid = fix_valid[fix_valid["FPOGD"] >= MIN_FIX_DUR]

    img_type_counts = fix_valid["img_type"].value_counts().to_dict()
    img_type_counts = {str(k): int(v) for k, v in img_type_counts.items()}

    # trial key for search/cross
    TRIAL_KEY = ["pid", "img_name", "tgt_id", "cue", "absent"]

    # -------------------------
    # (C) Build search trials (img_type=2) and cross trials (img_type=1)
    # -------------------------
    search_df = fix_valid[fix_valid["img_type"] == IMG_TYPE_SEARCH].copy()
    cross_df  = fix_valid[fix_valid["img_type"] == IMG_TYPE_CROSS].copy()

    # sort
    search_df = search_df.sort_values(TRIAL_KEY + ["TIME"])
    cross_df  = cross_df.sort_values(TRIAL_KEY + ["TIME"])

    # group keys
    search_keys = list(search_df.groupby(TRIAL_KEY, sort=False).groups.keys())
    cross_keys  = list(cross_df.groupby(TRIAL_KEY, sort=False).groups.keys())

    # sample for debug
    random.shuffle(search_keys)
    random.shuffle(cross_keys)
    search_debug_keys = search_keys[:SAMPLE_TRIALS_SEARCH_DEBUG]
    cross_debug_keys  = cross_keys[:SAMPLE_TRIALS_CROSS_DEBUG]

    # -------------------------
    # (D) Core evaluation per candidate
    #   D1: CROSS center-dispersion score (lower is better)
    #   D2: SEARCH target hit-rate (present trials only) (higher is better)
    # -------------------------
    cand_report = {}

    for cand in CANDIDATES:
        # ---- D1: CROSS center dispersion ----
        # compute distance to center in SEG space after mapping
        cross_dists = []
        cross_n = 0

        # We'll use seg/img for mapping; if missing seg/img, skip that trial.
        # For speed, sample a subset of cross trials (you can increase later)
        sample_cross_for_stats = cross_keys[:min(800, len(cross_keys))]

        for key in sample_cross_for_stats:
            g = cross_df.groupby(TRIAL_KEY, sort=False).get_group(key)
            if len(g) < MIN_FIX_PER_TRIAL:
                continue

            r0 = g.iloc[0]
            img_name = str(r0["img_name"])
            stem = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMG_DIR, img_name)
            seg_path = seg_idx.get(stem)

            if (not os.path.exists(img_path)) or (not seg_path) or (not os.path.exists(seg_path)):
                continue

            W_seg, H_seg = load_seg_shape(seg_path)
            orig_W = float(r0["original_width"])
            orig_H = float(r0["original_height"])

            cx = 0.5 * W_seg
            cy = 0.5 * H_seg

            for _, row in g.iterrows():
                vx = safe_float(row.get(cand.x_col))
                vy = safe_float(row.get(cand.y_col))
                if vx is None or vy is None:
                    continue
                fx = to_px_maybe(vx, orig_W) if cand.mode == "norm_or_px" else float(vx)
                fy = to_px_maybe(vy, orig_H) if cand.mode == "norm_or_px" else float(vy)
                sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
                sx = clamp(sx, 0, W_seg)
                sy = clamp(sy, 0, H_seg)

                d = float(np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2))
                cross_dists.append(d)
                cross_n += 1

        cross_stats = {}
        if cross_dists:
            arr = np.array(cross_dists, dtype=np.float64)
            cross_stats = {
                "n_points": int(len(arr)),
                "mean_dist_to_center_px": float(arr.mean()),
                "p50_dist_to_center_px": float(np.percentile(arr, 50)),
                "p90_dist_to_center_px": float(np.percentile(arr, 90)),
                "p99_dist_to_center_px": float(np.percentile(arr, 99)),
            }
        else:
            cross_stats = {"n_points": 0, "warning": "No cross points evaluated (missing seg/img?)"}

        # ---- D2: SEARCH target hit-rate (present only) ----
        # For search trials, compute per-fixation hit for each r in R_LIST in SEG space.
        # Only on absent==False. (absent==True should not be forced to hit.)
        search_hits = {str(r): {"any": 0, "dwell_min2": 0, "n_trials": 0} for r in R_LIST}
        sample_search_for_stats = search_keys[:min(1200, len(search_keys))]

        for key in sample_search_for_stats:
            pid, img_name, tgt_id, cue, absent = key
            if bool(absent):
                continue  # present-only validation

            g = search_df.groupby(TRIAL_KEY, sort=False).get_group(key)
            if len(g) < MIN_FIX_PER_TRIAL:
                continue

            r0 = g.iloc[0]
            img_name = str(r0["img_name"])
            stem = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMG_DIR, img_name)
            seg_path = seg_idx.get(stem)

            if (not os.path.exists(img_path)) or (not seg_path) or (not os.path.exists(seg_path)):
                continue

            W_seg, H_seg = load_seg_shape(seg_path)
            orig_W = float(r0["original_width"])
            orig_H = float(r0["original_height"])

            bbox_img = bbox_from_raw(r0, orig_W, orig_H)
            if bbox_img is None:
                continue

            # bbox -> seg
            x0, y0 = map_px_to_seg(bbox_img[0], bbox_img[1], orig_W, orig_H, W_seg, H_seg)
            x1, y1 = map_px_to_seg(bbox_img[2], bbox_img[3], orig_W, orig_H, W_seg, H_seg)
            x0 = clamp(x0, 0, W_seg); x1 = clamp(x1, 0, W_seg)
            y0 = clamp(y0, 0, H_seg); y1 = clamp(y1, 0, H_seg)
            bbox_seg = (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None
            if bbox_seg is None:
                continue

            # gather hits per r
            hits_by_r = {r: [] for r in R_LIST}

            for _, row in g.iterrows():
                vx = safe_float(row.get(cand.x_col))
                vy = safe_float(row.get(cand.y_col))
                if vx is None or vy is None:
                    continue

                fx = to_px_maybe(vx, orig_W) if cand.mode == "norm_or_px" else float(vx)
                fy = to_px_maybe(vy, orig_H) if cand.mode == "norm_or_px" else float(vy)
                sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
                sx = clamp(sx, 0, W_seg)
                sy = clamp(sy, 0, H_seg)

                for r in R_LIST:
                    bb = dilate_bbox_xyxy(bbox_seg, r, W_seg, H_seg)
                    if bb is None:
                        hits_by_r[r].append(0)
                    else:
                        hits_by_r[r].append(int(point_in_bbox(sx, sy, bb)))

            for r in R_LIST:
                h = hits_by_r[r]
                if not h:
                    continue
                any_hit = int(any(h))
                dwell2 = int(sum(h) >= 2)  # simple dwell criterion

                rr = str(r)
                search_hits[rr]["n_trials"] += 1
                search_hits[rr]["any"] += any_hit
                search_hits[rr]["dwell_min2"] += dwell2

        # finalize search rates
        search_rates = {}
        for rr, d in search_hits.items():
            n = max(1, d["n_trials"])
            search_rates[rr] = {
                "n_trials": int(d["n_trials"]),
                "any": int(d["any"]),
                "dwell_min2": int(d["dwell_min2"]),
                "any_rate": float(d["any"] / n),
                "dwell_min2_rate": float(d["dwell_min2"] / n),
            }

        cand_report[cand.name] = {
            "candidate": {"x_col": cand.x_col, "y_col": cand.y_col, "mode": cand.mode},
            "cross_center_dispersion": cross_stats,
            "search_target_hit_present_only": search_rates,
            "interpretation_hint": (
                "Better mapping usually has LOWER cross-center distances (img_type=1) "
                "and reasonably higher search hit-rate for moderate r (e.g., r=1.5 or r=2). "
                "Do NOT use img_type=3(validation) hit against search target."
            )
        }

    # -------------------------
    # (E) Choose a 'best' candidate (simple scoring)
    #   score = - mean_center_dist + 200 * dwell_rate(r=2.0)
    # -------------------------
    best_name = None
    best_score = -1e18
    scoring = {}

    for cname, rep in cand_report.items():
        cross_mean = rep["cross_center_dispersion"].get("mean_dist_to_center_px", None)
        r2 = rep["search_target_hit_present_only"].get("2.0", None)

        if cross_mean is None or r2 is None:
            score = -1e18
        else:
            score = (-float(cross_mean)) + 200.0 * float(r2.get("dwell_min2_rate", 0.0))
        scoring[cname] = float(score)
        if score > best_score:
            best_score = score
            best_name = cname

    # -------------------------
    # (F) Debug plots (using best candidate)
    # -------------------------
    debug_summary = {"search_debug": [], "cross_debug": []}

    if best_name is None:
        best_name = CANDIDATES[0].name

    best_cand = None
    for c in CANDIDATES:
        if c.name == best_name:
            best_cand = c
            break
    if best_cand is None:
        best_cand = CANDIDATES[0]

    # SEARCH debug
    search_grouped = search_df.groupby(TRIAL_KEY, sort=False)
    for i, key in enumerate(search_debug_keys):
        pid, img_name, tgt_id, cue, absent = key
        g = search_grouped.get_group(key)
        if len(g) < MIN_FIX_PER_TRIAL:
            continue

        r0 = g.iloc[0]
        img_name = str(r0["img_name"])
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        seg_path = seg_idx.get(stem)

        if (not os.path.exists(img_path)) or (not seg_path) or (not os.path.exists(seg_path)):
            continue

        W_seg, H_seg = load_seg_shape(seg_path)
        orig_W = float(r0["original_width"])
        orig_H = float(r0["original_height"])

        bbox_img = bbox_from_raw(r0, orig_W, orig_H)
        tgt_box_seg = None
        if bbox_img is not None:
            x0, y0 = map_px_to_seg(bbox_img[0], bbox_img[1], orig_W, orig_H, W_seg, H_seg)
            x1, y1 = map_px_to_seg(bbox_img[2], bbox_img[3], orig_W, orig_H, W_seg, H_seg)
            x0 = clamp(x0, 0, W_seg); x1 = clamp(x1, 0, W_seg)
            y0 = clamp(y0, 0, H_seg); y1 = clamp(y1, 0, H_seg)
            if x1 > x0 and y1 > y0:
                tgt_box_seg = (x0, y0, x1, y1)

        pts = []
        for _, row in g.iterrows():
            vx = safe_float(row.get(best_cand.x_col))
            vy = safe_float(row.get(best_cand.y_col))
            if vx is None or vy is None:
                continue
            fx = to_px_maybe(vx, orig_W) if best_cand.mode == "norm_or_px" else float(vx)
            fy = to_px_maybe(vy, orig_H) if best_cand.mode == "norm_or_px" else float(vy)
            sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
            pts.append((clamp(sx, 0, W_seg), clamp(sy, 0, H_seg)))
            if len(pts) >= MAX_POINTS_DRAW:
                break

        trial_id = trial_id_from_key(pid, img_name, tgt_id, cue, absent)
        t_min = float(g["TIME"].min())
        t_max = float(g["TIME"].max())
        dur = float(t_max - t_min)

        title = f"{trial_id} SEARCH  n_fix={len(g)}  dur={dur:.2f}s  cand={best_cand.name}"
        out_path = os.path.join(DEBUG_DIR, f"debug_search_{i:02d}_{stem}.png")
        render_debug_search(img_path, W_seg, H_seg, pts, tgt_box_seg, title, out_path)

        debug_summary["search_debug"].append({"trial_id": trial_id, "out_path": out_path})

    # CROSS debug
    cross_grouped = cross_df.groupby(TRIAL_KEY, sort=False)
    for i, key in enumerate(cross_debug_keys):
        pid, img_name, tgt_id, cue, absent = key
        g = cross_grouped.get_group(key)
        if len(g) < MIN_FIX_PER_TRIAL:
            continue

        r0 = g.iloc[0]
        img_name = str(r0["img_name"])
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMG_DIR, img_name)
        seg_path = seg_idx.get(stem)

        if (not os.path.exists(img_path)) or (not seg_path) or (not os.path.exists(seg_path)):
            continue

        W_seg, H_seg = load_seg_shape(seg_path)
        orig_W = float(r0["original_width"])
        orig_H = float(r0["original_height"])

        pts = []
        for _, row in g.iterrows():
            vx = safe_float(row.get(best_cand.x_col))
            vy = safe_float(row.get(best_cand.y_col))
            if vx is None or vy is None:
                continue
            fx = to_px_maybe(vx, orig_W) if best_cand.mode == "norm_or_px" else float(vx)
            fy = to_px_maybe(vy, orig_H) if best_cand.mode == "norm_or_px" else float(vy)
            sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
            pts.append((clamp(sx, 0, W_seg), clamp(sy, 0, H_seg)))
            if len(pts) >= MAX_POINTS_DRAW:
                break

        trial_id = trial_id_from_key(pid, img_name, tgt_id, cue, absent)
        t_min = float(g["TIME"].min())
        t_max = float(g["TIME"].max())
        dur = float(t_max - t_min)
        title = f"{trial_id} CROSS  n_fix={len(g)}  dur={dur:.2f}s  cand={best_cand.name}"
        out_path = os.path.join(DEBUG_DIR, f"debug_cross_{i:02d}_{stem}.png")
        render_debug_cross(img_path, W_seg, H_seg, pts, title, out_path)

        debug_summary["cross_debug"].append({"trial_id": trial_id, "out_path": out_path})

    # -------------------------
    # (G) Final report
    # -------------------------
    report = {
        "paths": {
            "BASE": BASE,
            "DATA_DIR": DATA_DIR,
            "FIX_CSV": FIX_CSV,
            "TGT_CSV": TGT_CSV,
            "IMG_DIR": IMG_DIR,
            "SEG_ROOT": SEG_ROOT,
        },
        "filters": {
            "validity": "FPOGV==1 & BPOGV==1",
            "min_fix_dur_s": float(MIN_FIX_DUR),
            "min_fix_per_trial": int(MIN_FIX_PER_TRIAL),
        },
        "join_check_fix_vs_targets": join_report,
        "img_type_counts_after_filters": img_type_counts,
        "candidates": cand_report,
        "scoring": {
            "rule": "score = -mean_cross_center_dist + 200*dwell_rate(r=2.0, present_only)",
            "scores": scoring,
            "best_candidate": best_name,
        },
        "debug": {
            "dir": DEBUG_DIR,
            "note": "Search debug shows trajectory + target bbox; Cross debug shows points + screen center.",
            "items": debug_summary,
        },
        "important_notes": [
            "Do NOT validate mapping by comparing img_type=3(validation) gaze against SEARCH target bbox. They are different tasks.",
            "The strongest sanity check is img_type=1(fixation cross): gaze should cluster near screen center after correct mapping.",
            "Search-phase target hit-rate should be computed on img_type=2 and preferably present-only (absent==False).",
            "If r=3 always 'wins', it may just reflect larger boxes increasing chance hits; prefer comparing moderate r (1.5/2.0).",
            "Fixations img_name is UEyes namespace; targets join must use ueyes_img_name."
        ]
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saved report:", OUT_REPORT)
    print("Debug images in:", DEBUG_DIR)
    print("Best candidate:", best_name)
    print("Join joined_pairs:", join_report["joined_pairs"], "join_rate:", round(join_report["join_rate_vs_fix_pairs"], 4))


if __name__ == "__main__":
    main()
