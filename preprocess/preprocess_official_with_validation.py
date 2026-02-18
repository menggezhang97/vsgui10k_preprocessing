import os
import json
import glob
import gzip
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# -------------------
# Paths (run from: C:\Users\zhang\GUI\VSGUI10K\data)
# -------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")

FIX_CSV  = os.path.join(DATA_DIR, "vsgui10k_fixations.csv")
IMG_DIR  = os.path.join(DATA_DIR, "vsgui10k-images")
SEG_ROOT = os.path.join(DATA_DIR, "segmentation")

OUT_JSONL = os.path.join(BASE, "trials_official_with_validation.jsonl.gz")
OUT_STATS = os.path.join(BASE, "preprocess_validation_stats.json")

DEBUG_DIR = os.path.join(BASE, "debug_official_with_validation")

# -------------------
# Official-ish settings
# -------------------
IMG_TYPE_TARGET = 0   # target description
IMG_TYPE_CROSS  = 1   # fixation cross
IMG_TYPE_SEARCH = 2   # visual search
IMG_TYPE_VALID  = 3   # selection/validation

# Fixation filtering (keep consistent with your current pipeline)
MIN_DUR = 0.08                # 80 ms
MIN_FIX_SEARCH = 3
MIN_FIX_VALID  = 1            # validation can be sparse but should exist often

# Official trial design: search max ~30s, validation ~3s (paper design)
# We won't hard-drop if >30s, but we will report outliers.
SEARCH_SOFT_MAX_S = 30.0
VALID_EXPECT_S = 3.0

# Validation success rules (simple, robust, deliverable):
# - For present trials (absent=0): consider validation as "hit" if ANY validation fixation falls
#   within expanded target bbox (r multipliers). Also report dwell(min_hits=2).
# - For absent trials: we can't verify absent button bbox because it's not in CSV;
#   we'll mark absent validation as "unknown" (but still store validation fixations).
R_MULTS = [1.0, 1.5, 2.0, 3.0]   # bbox expansion multipliers

DEBUG_N = 20
DEBUG_MAX_POINTS = 200
random.seed(7)

# Trial key used in your pipeline (works with your earlier check: phases [0,1,2,3] exist)
TRIAL_KEY = ["pid", "img_name", "tgt_id", "cue", "absent"]


# -------------------
# Helpers
# -------------------
def build_seg_index(seg_root: str) -> dict:
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


def load_seg_shape(seg_path: str):
    with open(seg_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    H, W, _ = d["img"]["shape"]
    return int(W), int(H)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def map_px_to_seg(x_px, y_px, orig_W, orig_H, W_seg, H_seg):
    rx = W_seg / float(orig_W)
    ry = H_seg / float(orig_H)
    return float(x_px * rx), float(y_px * ry)


def to_px(v, full):
    """
    Convert v to pixels given full dimension.
    If v looks normalized (<=1.5), treat as normalized and scale by full.
    Otherwise treat as already in pixels.
    """
    v = float(v)
    if 0.0 <= v <= 1.5:
        return v * float(full)
    return v


def expand_bbox(x0, y0, x1, y1, r, W, H):
    """
    Expand bbox around center by multiplier r.
    r=1 -> same bbox; r=2 -> width/height doubled.
    """
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = (x1 - x0) * r
    h = (y1 - y0) * r
    nx0 = clamp(cx - 0.5 * w, 0, W)
    nx1 = clamp(cx + 0.5 * w, 0, W)
    ny0 = clamp(cy - 0.5 * h, 0, H)
    ny1 = clamp(cy + 0.5 * h, 0, H)
    return nx0, ny0, nx1, ny1


def point_in_bbox(x, y, x0, y0, x1, y1):
    return (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)


def render_debug(img_path, W_seg, H_seg, fix_xy_seg, tgt_box_seg, out_path, title, draw_line=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_seg, H_seg), Image.BILINEAR)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)

    # target bbox
    if tgt_box_seg is not None:
        x0, y0, x1, y1 = tgt_box_seg
        plt.plot([x0, x1, x1, x0, x0],
                 [y0, y0, y1, y1, y0],
                 linewidth=3, color="red")

    if fix_xy_seg:
        xs = [p[0] for p in fix_xy_seg]
        ys = [p[1] for p in fix_xy_seg]

        if draw_line and len(xs) >= 2:
            plt.plot(xs, ys, linewidth=2, color="cyan")

        plt.scatter(xs, ys, s=55, marker="o",
                    c="yellow", edgecolors="black", linewidths=0.7)

        for i, (x, y) in enumerate(fix_xy_seg, start=1):
            plt.text(x + 4, y - 4, str(i),
                     fontsize=9, color="black",
                     bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

    plt.xlim([0, W_seg])
    plt.ylim([H_seg, 0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------
# Main
# -------------------
def main():
    assert os.path.exists(FIX_CSV), f"Missing: {FIX_CSV}"
    assert os.path.isdir(IMG_DIR), f"Missing: {IMG_DIR}"
    assert os.path.isdir(SEG_ROOT), f"Missing: {SEG_ROOT}"

    os.makedirs(DEBUG_DIR, exist_ok=True)

    seg_idx = build_seg_index(SEG_ROOT)
    print("Seg index size:", len(seg_idx))

    df = pd.read_csv(FIX_CSV, low_memory=False)

    # Basic validity + duration filter (apply to all phases consistently)
    # Note: the paper says gaze stream/fixation data is preprocessed; filtering unrealistic fixations is reasonable :contentReference[oaicite:2]{index=2}
    df = df[(df["FPOGV"] == 1) & (df["BPOGV"] == 1)]
    df = df[df["FPOGD"] >= MIN_DUR]

    # Sort stable
    df = df.sort_values(TRIAL_KEY + ["img_type", "TIME"])

    # Stats collectors
    stats = {
        "n_rows_after_filters": int(len(df)),
        "img_type_counts": Counter(df["img_type"].tolist()),
        "kept_trials": 0,
        "dropped_missing_img_or_seg": 0,
        "dropped_no_search_phase": 0,
        "dropped_search_too_short": 0,
        "dropped_no_validation_phase": 0,
        "search_dur_outliers_gt30": 0,
        "validation_dur_stats": {},
        "search_dur_stats": {},
        "coord_oob_counts": {
            "search": {"raw_scaled_outside_01": 0, "n_points": 0},
            "valid":  {"raw_scaled_outside_01": 0, "n_points": 0},
        },
        "validation_success_present": {f"r{r}".replace(".","_"): {"any": 0, "dwell_min2": 0, "n": 0} for r in R_MULTS},
        "validation_absent_trials": 0,
    }

    debug_done = 0

    with gzip.open(OUT_JSONL, "wt", encoding="utf-8") as f_out:
        for (pid, img_name, tgt_id, cue, absent), g_all in df.groupby(TRIAL_KEY, sort=False):

            img_name = str(img_name)
            stem = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMG_DIR, img_name)
            seg_path = seg_idx.get(stem)

            if (not os.path.exists(img_path)) or (not seg_path) or (not os.path.exists(seg_path)):
                stats["dropped_missing_img_or_seg"] += 1
                continue

            # Split phases
            g_search = g_all[g_all["img_type"] == IMG_TYPE_SEARCH]
            g_valid  = g_all[g_all["img_type"] == IMG_TYPE_VALID]

            if len(g_search) == 0:
                stats["dropped_no_search_phase"] += 1
                continue
            if len(g_search) < MIN_FIX_SEARCH:
                stats["dropped_search_too_short"] += 1
                continue
            if len(g_valid) == 0:
                stats["dropped_no_validation_phase"] += 1
                # We still keep the trial (some recordings might be missing),
                # but we mark validation as missing.
                # If you prefer strict official: change to "continue".
                # For a deliverable pipeline, keeping with a flag is more useful.

            # Load seg geometry
            W_seg, H_seg = load_seg_shape(seg_path)

            # Screenshot original size (from CSV)
            r0 = g_search.iloc[0]
            orig_W = float(r0["original_width"])
            orig_H = float(r0["original_height"])

            # Target bbox (from CSV; can be normalized or px)
            raw_tx = float(r0["tgt_x"]); raw_ty = float(r0["tgt_y"])
            raw_tw = float(r0["tgt_width"]); raw_th = float(r0["tgt_height"])
            tx = to_px(raw_tx, orig_W); ty = to_px(raw_ty, orig_H)
            tw = to_px(raw_tw, orig_W); th = to_px(raw_th, orig_H)

            x0, y0 = map_px_to_seg(tx, ty, orig_W, orig_H, W_seg, H_seg)
            x1, y1 = map_px_to_seg(tx + tw, ty + th, orig_W, orig_H, W_seg, H_seg)
            x0 = clamp(x0, 0, W_seg); x1 = clamp(x1, 0, W_seg)
            y0 = clamp(y0, 0, H_seg); y1 = clamp(y1, 0, H_seg)
            tgt_box_seg = (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

            # Search duration
            t_min_s = float(g_search["TIME"].min())
            t_max_s = float(g_search["TIME"].max())
            search_dur = float(t_max_s - t_min_s)
            if search_dur > SEARCH_SOFT_MAX_S + 1e-6:
                stats["search_dur_outliers_gt30"] += 1

            # Validation duration
            valid_dur = None
            if len(g_valid) > 0:
                tv0 = float(g_valid["TIME"].min())
                tv1 = float(g_valid["TIME"].max())
                valid_dur = float(tv1 - tv0)

            # Build scanpath for search
            search_scanpath = []
            search_xy_seg_for_debug = []
            for i, (_, row) in enumerate(g_search.iterrows()):
                raw_fx = float(row["FPOGX_scaled"])
                raw_fy = float(row["FPOGY_scaled"])

                stats["coord_oob_counts"]["search"]["n_points"] += 1
                if (raw_fx < 0.0) or (raw_fx > 1.0) or (raw_fy < 0.0) or (raw_fy > 1.0):
                    stats["coord_oob_counts"]["search"]["raw_scaled_outside_01"] += 1

                fx = to_px(raw_fx, orig_W)
                fy = to_px(raw_fy, orig_H)

                sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
                sx = clamp(sx, 0, W_seg)
                sy = clamp(sy, 0, H_seg)

                dur = float(row["FPOGD"])
                t_rel = float(float(row["TIME"]) - t_min_s)

                search_scanpath.append({
                    "i": int(i),
                    "t": float(t_rel),
                    "dur": float(dur),
                    "xy_seg": {"x": float(sx), "y": float(sy)},
                    "xy_img_px": {"x": float(fx), "y": float(fy)},
                    "xy_img_raw_scaled": {"x": float(raw_fx), "y": float(raw_fy)},
                })

                if len(search_xy_seg_for_debug) < DEBUG_MAX_POINTS:
                    search_xy_seg_for_debug.append((sx, sy))

            # Build validation fixations (for success check)
            valid_fix = []
            valid_xy_seg_for_debug = []
            valid_t0 = None
            if len(g_valid) > 0:
                valid_t0 = float(g_valid["TIME"].min())
                for j, (_, row) in enumerate(g_valid.iterrows()):
                    raw_fx = float(row["FPOGX_scaled"])
                    raw_fy = float(row["FPOGY_scaled"])

                    stats["coord_oob_counts"]["valid"]["n_points"] += 1
                    if (raw_fx < 0.0) or (raw_fx > 1.0) or (raw_fy < 0.0) or (raw_fy > 1.0):
                        stats["coord_oob_counts"]["valid"]["raw_scaled_outside_01"] += 1

                    fx = to_px(raw_fx, orig_W)
                    fy = to_px(raw_fy, orig_H)
                    sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
                    sx = clamp(sx, 0, W_seg)
                    sy = clamp(sy, 0, H_seg)

                    t_rel = float(float(row["TIME"]) - valid_t0)
                    dur = float(row["FPOGD"])

                    valid_fix.append({
                        "j": int(j),
                        "t": float(t_rel),
                        "dur": float(dur),
                        "xy_seg": {"x": float(sx), "y": float(sy)},
                        "xy_img_px": {"x": float(fx), "y": float(fy)},
                        "xy_img_raw_scaled": {"x": float(raw_fx), "y": float(raw_fy)},
                    })

                    if len(valid_xy_seg_for_debug) < DEBUG_MAX_POINTS:
                        valid_xy_seg_for_debug.append((sx, sy))

            # -------------------
            # Validation success (present trials only)
            # -------------------
            absent_flag = int(bool(absent))
            success = {
                "present_trial": int(1 - absent_flag),
                "validation_available": int(1 if len(valid_fix) > 0 else 0),
                "rules": {},
            }

            if absent_flag == 1:
                stats["validation_absent_trials"] += 1
                # Without absent-button bbox, we cannot verify correctness robustly.
                # Keep it explicit.
                for r in R_MULTS:
                    key = f"r{str(r).replace('.','_')}"
                    success["rules"][key] = {"any": None, "dwell_min2": None}
            else:
                # present trial: can verify against target bbox
                if tgt_box_seg is None or len(valid_fix) == 0:
                    for r in R_MULTS:
                        key = f"r{str(r).replace('.','_')}"
                        success["rules"][key] = {"any": 0, "dwell_min2": 0}
                        stats["validation_success_present"][key]["n"] += 1
                else:
                    x0b, y0b, x1b, y1b = tgt_box_seg
                    # For dwell, count number of validation fixations inside bbox
                    pts = [(v["xy_seg"]["x"], v["xy_seg"]["y"]) for v in valid_fix]

                    for r in R_MULTS:
                        key = f"r{str(r).replace('.','_')}"
                        ex0, ey0, ex1, ey1 = expand_bbox(x0b, y0b, x1b, y1b, r, W_seg, H_seg)
                        inside = [point_in_bbox(px, py, ex0, ey0, ex1, ey1) for (px, py) in pts]
                        n_inside = int(sum(inside))

                        any_hit = 1 if n_inside >= 1 else 0
                        dwell_min2 = 1 if n_inside >= 2 else 0

                        success["rules"][key] = {"any": any_hit, "dwell_min2": dwell_min2}

                        stats["validation_success_present"][key]["n"] += 1
                        stats["validation_success_present"][key]["any"] += any_hit
                        stats["validation_success_present"][key]["dwell_min2"] += dwell_min2

            # Trial id
            trial_id = f"{pid}|{img_name}|{tgt_id}|{cue}|abs{absent_flag}"

            out = {
                "trial_id": trial_id,
                "key": {
                    "pid": str(pid),
                    "img_name": img_name,
                    "tgt_id": str(tgt_id),
                    "cue": str(cue),
                    "absent": int(bool(absent)),
                },
                "geom": {
                    "original_w": float(orig_W),
                    "original_h": float(orig_H),
                    "seg_w": int(W_seg),
                    "seg_h": int(H_seg),
                },
                "target": {
                    "bbox_img_raw": {"x": raw_tx, "y": raw_ty, "w": raw_tw, "h": raw_th},
                    "bbox_img_px": {"x": tx, "y": ty, "w": tw, "h": th},
                    "bbox_seg": None if tgt_box_seg is None else {
                        "x0": float(tgt_box_seg[0]),
                        "y0": float(tgt_box_seg[1]),
                        "x1": float(tgt_box_seg[2]),
                        "y1": float(tgt_box_seg[3]),
                    }
                },
                "phases": {
                    "search": {
                        "img_type": IMG_TYPE_SEARCH,
                        "n_fix": int(len(search_scanpath)),
                        "t_end": float(search_dur),
                        "scanpath": search_scanpath,
                    },
                    "validation": {
                        "img_type": IMG_TYPE_VALID,
                        "available": int(1 if len(valid_fix) > 0 else 0),
                        "n_fix": int(len(valid_fix)),
                        "t_end": None if valid_dur is None else float(valid_dur),
                        "fixations": valid_fix,
                    }
                },
                "success": success,
                "meta": {
                    "seg_path": seg_path,
                    "filters": {
                        "validity": "FPOGV==1 & BPOGV==1",
                        "min_dur": MIN_DUR,
                        "min_fix_search": MIN_FIX_SEARCH,
                        "min_fix_valid": MIN_FIX_VALID,
                    },
                    "official_notes": {
                        "design": "Search screen max ~30s; validation screen ~3s, used to validate found location.",
                        "source_pdf": "visual-search-GUI-pre-registration",
                    }
                }
            }

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["kept_trials"] += 1

            # Debug images
            if debug_done < DEBUG_N:
                title_s = f"{trial_id} SEARCH  n_fix={len(search_scanpath)}  dur={search_dur:.2f}s"
                out_s = os.path.join(DEBUG_DIR, f"debug_search_{debug_done:02d}_{stem}.png")
                render_debug(img_path, W_seg, H_seg, search_xy_seg_for_debug, tgt_box_seg, out_s, title_s, draw_line=True)

                title_v = f"{trial_id} VALID  n_fix={len(valid_fix)}  dur={(valid_dur if valid_dur is not None else -1):.2f}s"
                out_v = os.path.join(DEBUG_DIR, f"debug_validation_{debug_done:02d}_{stem}.png")
                render_debug(img_path, W_seg, H_seg, valid_xy_seg_for_debug, tgt_box_seg, out_v, title_v, draw_line=False)

                debug_done += 1

    # Summarize duration stats
    # Search
    # Note: we only computed search_dur per trial; easiest is to recompute quickly from kept jsonl,
    # but for a deliverable, we approximate using aggregated inside loop is more work.
    # We'll read back first N lines for summary, and also report outliers count.
    # (Lightweight, OK.)
    search_durs = []
    valid_durs = []
    with gzip.open(OUT_JSONL, "rt", encoding="utf-8") as f_in:
        for line in f_in:
            d = json.loads(line)
            sd = d["phases"]["search"]["t_end"]
            search_durs.append(sd)
            vd = d["phases"]["validation"]["t_end"]
            if vd is not None:
                valid_durs.append(vd)

    def _dur_stats(arr):
        if len(arr) == 0:
            return {}
        arr = np.array(arr, dtype=float)
        return {
            "n": int(len(arr)),
            "min": float(np.min(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    stats["search_dur_stats"] = _dur_stats(search_durs)
    stats["validation_dur_stats"] = _dur_stats(valid_durs)

    # Convert Counters to dict
    stats["img_type_counts"] = {str(k): int(v) for k, v in stats["img_type_counts"].items()}

    # Convert success stats to rates
    for k, v in stats["validation_success_present"].items():
        n = v["n"]
        if n > 0:
            v["any_rate"] = float(v["any"]) / float(n)
            v["dwell_min2_rate"] = float(v["dwell_min2"]) / float(n)

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved:", OUT_JSONL)
    print("Saved stats:", OUT_STATS)
    print("Debug images in:", DEBUG_DIR)
    print("Kept trials:", stats["kept_trials"])
    print("Dropped missing img/seg:", stats["dropped_missing_img_or_seg"])
    print("Dropped no search phase:", stats["dropped_no_search_phase"])
    print("Dropped short search:", stats["dropped_search_too_short"])
    print("Trials missing validation phase (kept with flag):", stats["dropped_no_validation_phase"])
    print("Search dur outliers >30s:", stats["search_dur_outliers_gt30"])


if __name__ == "__main__":
    main()
