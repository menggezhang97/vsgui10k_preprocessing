import os
import json
import glob
import random
import gzip
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# -------------------
# Config
# -------------------
FIX_CSV = r"data\vsgui10k_fixations.csv"
IMG_DIR = r"data\vsgui10k-images"
SEG_ROOT = r"data\segmentation"

OUT_JSONL = "trials_with_elements.jsonl.gz"
OUT_STATS = "preprocess_stats.json"
DEBUG_DIR = "debug_examples"

# Filtering
MIN_DUR = 0.08
MIN_FIX_PER_TRIAL = 3

# RAW gaze coords may slightly go outside [0,1]
ALLOW_MIN = -0.10
ALLOW_MAX = 1.10

# Success / evaluation params (MAX FLEXIBILITY)
R_LIST = [1.0, 1.5, 2.0, 3.0]     # bbox dilation ratios
K_LIST = [1, 3, 5, 10]           # lastK windows (by fixation count)
T_LIST = [0.5, 1.0, 2.0]         # lastT windows (by seconds)
DWELL_MIN_HITS = 2               # verification: at least N hits within target neighborhood
PRIMARY_R_FOR_FOUND = 2.0        # default r for found time / confusion (still keep all r in JSON)

# Debug
DEBUG_N = 12
DEBUG_DRAW_BOXES_MAX = 40

TRIAL_KEY = ["pid", "img_name", "tgt_id", "cue", "absent"]

random.seed(7)


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


def load_segmentation(seg_path: str):
    with open(seg_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    H, W, _ = d["img"]["shape"]
    compos = d.get("compos", [])

    boxes = []
    for i, c in enumerate(compos):
        try:
            x1 = int(c["column_min"]); x2 = int(c["column_max"])
            y1 = int(c["row_min"]);    y2 = int(c["row_max"])
            cls = str(c.get("class", ""))
        except Exception:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        boxes.append({
            "id": f"c_{i:05d}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cls": cls, "area": area
        })

    return W, H, boxes


def point_in_box(x, y, b):
    return (b["x1"] <= x <= b["x2"]) and (b["y1"] <= y <= b["y2"])


def map_fix_to_element(x_px: float, y_px: float, boxes: list):
    inside = [b for b in boxes if point_in_box(x_px, y_px, b)]
    if inside:
        best = min(inside, key=lambda b: b["area"])
        return {
            "id": best["id"],
            "class": best["cls"],
            "n_boxes_hit": int(len(inside)),
            "ambiguous": int(len(inside) > 1),
            "hit_rule": "smallest_area"
        }
    return {
        "id": None,
        "class": "Background",
        "n_boxes_hit": 0,
        "ambiguous": 0,
        "hit_rule": "smallest_area"
    }


def clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def compute_target_box_in_seg_pixels(r0, W_seg, H_seg):
    """
    Target bbox mapping:
      tgt_x, tgt_y, tgt_width, tgt_height: normalized in ORIGINAL stimulus frame
      original_width, original_height
      scale_x, scale_y
    Map: original norm -> original px -> scaled/displayed px -> seg px (ratio)
    """
    tx = safe_float(r0.get("tgt_x"))
    ty = safe_float(r0.get("tgt_y"))
    tw = safe_float(r0.get("tgt_width"))
    th = safe_float(r0.get("tgt_height"))

    orig_W = safe_float(r0.get("original_width"))
    orig_H = safe_float(r0.get("original_height"))
    sx = safe_float(r0.get("scale_x"))
    sy = safe_float(r0.get("scale_y"))

    if None in (tx, ty, tw, th, orig_W, orig_H, sx, sy):
        return None
    if orig_W <= 0 or orig_H <= 0 or sx <= 0 or sy <= 0:
        return None

    # 1) normalized -> original pixels
    x1_o = tx * orig_W
    y1_o = ty * orig_H
    w_o  = tw * orig_W
    h_o  = th * orig_H

    # 2) original pixels -> displayed/scaled pixels
    x1_s = x1_o * sx
    y1_s = y1_o * sy
    w_s  = w_o  * sx
    h_s  = h_o  * sy

    disp_W = orig_W * sx
    disp_H = orig_H * sy
    if disp_W <= 0 or disp_H <= 0:
        return None

    # 3) displayed pixels -> segmentation pixels
    rx = W_seg / disp_W
    ry = H_seg / disp_H

    x1 = x1_s * rx
    y1 = y1_s * ry
    x2 = (x1_s + w_s) * rx
    y2 = (y1_s + h_s) * ry

    # clamp
    x1 = max(0.0, min(float(W_seg), x1))
    x2 = max(0.0, min(float(W_seg), x2))
    y1 = max(0.0, min(float(H_seg), y1))
    y2 = max(0.0, min(float(H_seg), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def bbox_center_diag_area(b):
    x1, y1, x2, y2 = b
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    diag = float(np.sqrt(w * w + h * h))
    area = float(w * h)
    return cx, cy, diag, area


def dilate_bbox(b, r, W_seg, H_seg):
    x1, y1, x2, y2 = b
    cx, cy, _, _ = bbox_center_diag_area(b)
    w = (x2 - x1) * r
    h = (y2 - y1) * r
    nx1 = cx - 0.5 * w
    ny1 = cy - 0.5 * h
    nx2 = cx + 0.5 * w
    ny2 = cy + 0.5 * h

    # clamp
    nx1 = max(0.0, min(float(W_seg), nx1))
    nx2 = max(0.0, min(float(W_seg), nx2))
    ny1 = max(0.0, min(float(H_seg), ny1))
    ny2 = max(0.0, min(float(H_seg), ny2))

    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def point_in_bbox_xy(x, y, b):
    x1, y1, x2, y2 = b
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def dist_point_to_bbox(x, y, b):
    x1, y1, x2, y2 = b
    dx = 0.0
    if x < x1:
        dx = x1 - x
    elif x > x2:
        dx = x - x2
    dy = 0.0
    if y < y1:
        dy = y1 - y
    elif y > y2:
        dy = y - y2
    return float(np.sqrt(dx * dx + dy * dy))


def r_key(r: float) -> str:
    if abs(r - round(r)) < 1e-9:
        return f"r{int(round(r))}"
    s = str(r).replace(".", "_")
    return f"r{s}"


def compute_lastK(hit_list, K):
    if K <= 0:
        return {"any": 0, "ratio": 0.0, "m2": 0}
    tail = hit_list[-K:] if len(hit_list) >= K else hit_list[:]
    if not tail:
        return {"any": 0, "ratio": 0.0, "m2": 0}
    s = int(sum(tail))
    return {"any": int(s > 0), "ratio": float(s / len(tail)), "m2": int(s >= 2)}


def compute_lastT(t_rel_list, hit_list, T, search_dur):
    if T <= 0 or search_dur <= 0:
        return {"any": 0, "ratio": 0.0, "m2": 0}
    start_t = max(0.0, search_dur - T)
    idx = [i for i, t in enumerate(t_rel_list) if t >= start_t]
    if not idx:
        return {"any": 0, "ratio": 0.0, "m2": 0}
    tail = [hit_list[i] for i in idx]
    s = int(sum(tail))
    return {"any": int(s > 0), "ratio": float(s / len(tail)), "m2": int(s >= 2)}


def find_first_dwell_hit_index(hit_list, min_hits):
    c = 0
    for i, h in enumerate(hit_list):
        if h:
            c += 1
            if c >= min_hits:
                return i
    return None


def render_debug(img_path, boxes, fix_pts_seg, fix_flags, tgt_box_seg, out_path, title, W_seg, H_seg):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_seg, H_seg), Image.BILINEAR)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)

    drawn = 0
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        plt.plot([x1, x2, x2, x1, x1],
                 [y1, y1, y2, y2, y1],
                 linewidth=1)
        drawn += 1
        if drawn >= DEBUG_DRAW_BOXES_MAX:
            break

    if tgt_box_seg is not None:
        x1, y1, x2, y2 = tgt_box_seg
        plt.plot([x1, x2, x2, x1, x1],
                 [y1, y1, y2, y2, y1],
                 linewidth=3)

    xs1, ys1, xs0, ys0 = [], [], [], []
    for (x, y), flag in zip(fix_pts_seg, fix_flags):
        if flag:
            xs1.append(x); ys1.append(y)
        else:
            xs0.append(x); ys0.append(y)

    if xs0:
        plt.scatter(xs0, ys0, s=18, marker="o")
    if xs1:
        plt.scatter(xs1, ys1, s=26, marker="x")

    plt.xlim([0, W_seg])
    plt.ylim([H_seg, 0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------
# Main
# -------------------
def main():
    seg_idx = build_seg_index(SEG_ROOT)
    print("Seg index size:", len(seg_idx))

    df = pd.read_csv(FIX_CSV)

    stats = {
        "rows_raw": int(len(df)),
        "rows_after_validity": None,
        "rows_after_duration": None,
        "rows_after_allow_range": None,
        "rows_clipped_to_01": None,

        "trials_raw": None,
        "trials_kept": None,
        "trials_dropped_too_short": 0,
        "trials_dropped_missing_seg": 0,
        "missing_seg_examples": [],

        "element_class_counts": {},
        "background_ratio": None,
        "hit_ratio_non_background": None,

        "n_fix_per_trial_summary": {},

        "params": {
            "R_LIST": R_LIST,
            "K_LIST": K_LIST,
            "T_LIST": T_LIST,
            "DWELL_MIN_HITS": DWELL_MIN_HITS,
            "PRIMARY_R_FOR_FOUND": PRIMARY_R_FOR_FOUND,
            "MIN_DUR": MIN_DUR,
            "MIN_FIX_PER_TRIAL": MIN_FIX_PER_TRIAL,
            "ALLOW_RANGE": [ALLOW_MIN, ALLOW_MAX]
        }
    }

    # 1) Validity filter
    df = df[(df["FPOGV"] == 1) & (df["BPOGV"] == 1)]
    stats["rows_after_validity"] = int(len(df))

    # 2) Duration filter
    df = df[df["FPOGD"] >= MIN_DUR]
    stats["rows_after_duration"] = int(len(df))

    # 3) Allow mild out-of-range
    df = df[(df["FPOGX"] >= ALLOW_MIN) & (df["FPOGX"] <= ALLOW_MAX) &
            (df["FPOGY"] >= ALLOW_MIN) & (df["FPOGY"] <= ALLOW_MAX)]
    stats["rows_after_allow_range"] = int(len(df))

    # 4) Sort within trial
    df = df.sort_values(TRIAL_KEY + ["TIME"])
    stats["trials_raw"] = int(len(df[TRIAL_KEY].drop_duplicates()))

    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_done = 0

    elem_counter = Counter()
    nfix_list = []

    total_fix = 0
    total_bg = 0
    total_hit = 0
    clipped_count = 0

    # Precompute primary r index (for debug + found time rule selection)
    j_primary = R_LIST.index(PRIMARY_R_FOR_FOUND) if PRIMARY_R_FOR_FOUND in R_LIST else 0
    rk_primary = r_key(PRIMARY_R_FOR_FOUND)

    with gzip.open(OUT_JSONL, "wt", encoding="utf-8") as f_out:
        for (pid, img_name, tgt_id, cue, absent), g in df.groupby(TRIAL_KEY, sort=False):

            if g["img_name"].nunique() != 1:
                raise RuntimeError("Trial grouping wrong: multiple img_name in one trial")

            if len(g) < MIN_FIX_PER_TRIAL:
                stats["trials_dropped_too_short"] += 1
                continue

            r0 = g.iloc[0]
            media_id = int(r0["media_id"]) if "media_id" in g.columns else None

            img_name = str(img_name)
            img_stem = os.path.splitext(img_name)[0]
            img_path = os.path.join(IMG_DIR, img_name)

            seg_path = seg_idx.get(img_stem)
            if not seg_path or not os.path.exists(seg_path):
                stats["trials_dropped_missing_seg"] += 1
                if len(stats["missing_seg_examples"]) < 10:
                    stats["missing_seg_examples"].append(img_name)
                continue

            W_seg, H_seg, boxes = load_segmentation(seg_path)

            # target bbox in seg pixels
            tgt_box_seg = compute_target_box_in_seg_pixels(r0, W_seg, H_seg)

            tgt_center = None
            tgt_diag = None
            tgt_area = None
            if tgt_box_seg is not None:
                cx, cy, diag, area = bbox_center_diag_area(tgt_box_seg)
                tgt_center = {"x": float(cx), "y": float(cy)}
                tgt_diag = float(diag)
                tgt_area = float(area)

            # time base
            t_min = float(g["TIME"].min())
            t_max = float(g["TIME"].max())
            search_dur = float(t_max - t_min)

            # precompute dilated bboxes for each r (keyed by r_key)
            dilated = {}
            if tgt_box_seg is not None:
                for r in R_LIST:
                    dilated[r_key(r)] = dilate_bbox(tgt_box_seg, r, W_seg, H_seg)
            else:
                for r in R_LIST:
                    dilated[r_key(r)] = None

            scanpath = []
            fix_pts_for_debug = []
            debug_flag_for_points = []

            t_rel_list = []
            # list-of-lists aligned with R_LIST: in_target_lists[j][i_fix]
            in_target_lists = [[] for _ in R_LIST]

            prev_t_rel = None

            # iterate faster
            for i, row in enumerate(g.itertuples(index=False)):
                x_raw = float(getattr(row, "FPOGX"))
                y_raw = float(getattr(row, "FPOGY"))

                x = clip01(x_raw)
                y = clip01(y_raw)
                if (x != x_raw) or (y != y_raw):
                    clipped_count += 1

                dur = float(getattr(row, "FPOGD"))

                t_abs = float(getattr(row, "TIME"))
                t_rel = float(t_abs - t_min)
                t_end = float(t_rel + dur)
                dt = None if prev_t_rel is None else float(t_rel - prev_t_rel)
                prev_t_rel = t_rel

                progress_t = float(t_rel / search_dur) if search_dur > 0 else 0.0
                progress_i = float(i / (len(g) - 1)) if len(g) > 1 else 0.0

                x_px = float(x * W_seg)
                y_px = float(y * H_seg)

                elem = map_fix_to_element(x_px, y_px, boxes)

                total_fix += 1
                elem_counter[elem["class"]] += 1
                if elem["class"] == "Background":
                    total_bg += 1
                else:
                    total_hit += 1

                # --- target metrics: in_target as LIST aligned with R_LIST ---
                dist = None
                if tgt_box_seg is not None:
                    dist = dist_point_to_bbox(x_px, y_px, tgt_box_seg)

                    in_target_vec = []
                    for j, r in enumerate(R_LIST):
                        bb = dilated[r_key(r)]
                        v = int(bb is not None and point_in_bbox_xy(x_px, y_px, bb))
                        in_target_vec.append(v)
                        in_target_lists[j].append(v)
                else:
                    in_target_vec = [0] * len(R_LIST)
                    for j in range(len(R_LIST)):
                        in_target_lists[j].append(0)

                t_rel_list.append(t_rel)

                scanpath.append({
                    "i": int(i),

                    # time
                    "t": float(t_rel),
                    "dur": float(dur),
                    "dt": None if dt is None else float(dt),
                    "t_end": float(t_end),
                    "progress_t": float(progress_t),
                    "progress_i": float(progress_i),

                    # coords
                    "xy_norm": {"x": float(x), "y": float(y)},
                    "xy_norm_raw": {"x": float(x_raw), "y": float(y_raw)},
                    "xy_seg": {"x": float(x_px), "y": float(y_px)},

                    # UI element mapping
                    "elem": elem,

                    # target relationship
                    "hit": {
                        "dist_to_target": None if dist is None else float(dist),
                        "in_target": in_target_vec  # list aligned with params.r_list
                    }
                })

                if len(fix_pts_for_debug) < 160:
                    fix_pts_for_debug.append((x_px, y_px))
                    debug_flag_for_points.append(bool(in_target_vec[j_primary]))

            # -------- trial-level success summaries --------
            success_any = {}
            success_dwell = {"min_hits": int(DWELL_MIN_HITS)}
            success_lastK = {}
            success_lastT = {}

            for j, r in enumerate(R_LIST):
                rk = r_key(r)
                hits = in_target_lists[j]

                success_any[rk] = int(any(hits))
                success_dwell[rk] = int(sum(hits) >= DWELL_MIN_HITS)

                lk = {}
                for K in K_LIST:
                    lk[f"K{K}"] = compute_lastK(hits, K)
                success_lastK[rk] = lk

                lt = {}
                for T in T_LIST:
                    lt[f"T{str(T).replace('.','_')}"] = compute_lastT(t_rel_list, hits, T, search_dur)
                success_lastT[rk] = lt

            # found time based on PRIMARY_R_FOR_FOUND and dwell criterion
            hits_primary = in_target_lists[j_primary]
            idx_found = find_first_dwell_hit_index(hits_primary, DWELL_MIN_HITS)
            t_found = None
            time_to_target = None
            if idx_found is not None:
                t_found = float(scanpath[idx_found]["t"])
                time_to_target = float(t_found)

            # build stable trial_id
            trial_id = f"{pid}|{img_name}|{tgt_id}|{cue}|abs{int(bool(absent))}"

            trial = {
                "trial_id": trial_id,
                "key": {
                    "pid": str(pid),
                    "img_name": str(img_name),
                    "tgt_id": int(tgt_id) if str(tgt_id).isdigit() else str(tgt_id),
                    "cue": str(cue),
                    "absent": int(bool(absent))
                },
                "meta": {
                    "media_id": media_id,
                    "category": str(r0.get("category", "")),
                    "n_fix": int(len(scanpath)),
                    "t_start": 0.0,
                    "t_end": float(search_dur),
                    "search_dur": float(search_dur),
                    "seg_shape": {"w": int(W_seg), "h": int(H_seg)},
                    "coord_src": "FPOGX/FPOGY_raw",
                    "filter": {
                        "validity": "FPOGV==1 & BPOGV==1",
                        "min_dur": float(MIN_DUR),
                        "oob_allow": [float(ALLOW_MIN), float(ALLOW_MAX)],
                        "clip_to_01": True
                    }
                },
                "params": {
                    "r_list": [float(r) for r in R_LIST],
                    "K_list": [int(k) for k in K_LIST],
                    "T_list": [float(t) for t in T_LIST],
                    "dwell_min_hits": int(DWELL_MIN_HITS),
                    "primary_r_for_found": float(PRIMARY_R_FOR_FOUND)
                },
                "task": {
                    "absent": int(bool(absent)),
                    "cue": str(cue),
                    "tgt_id": int(tgt_id) if str(tgt_id).isdigit() else str(tgt_id)
                },
                "target": {
                    # original fields
                    "tgt_x": float(r0.get("tgt_x", float("nan"))),
                    "tgt_y": float(r0.get("tgt_y", float("nan"))),
                    "tgt_w": float(r0.get("tgt_width", float("nan"))),
                    "tgt_h": float(r0.get("tgt_height", float("nan"))),
                    "tgt_color": str(r0.get("tgt_color", "")),
                    "tgt_text": str(r0.get("tgt_text", "")),
                    "tgt_location": str(r0.get("tgt_location", "")),
                    "original_width": float(r0.get("original_width", float("nan"))),
                    "original_height": float(r0.get("original_height", float("nan"))),
                    "scale_x": float(r0.get("scale_x", float("nan"))),
                    "scale_y": float(r0.get("scale_y", float("nan"))),

                    # seg-space geometry
                    "bbox_seg": None if tgt_box_seg is None else {
                        "x0": float(tgt_box_seg[0]),
                        "y0": float(tgt_box_seg[1]),
                        "x1": float(tgt_box_seg[2]),
                        "y1": float(tgt_box_seg[3]),
                    },
                    "center_seg": tgt_center,
                    "diag_seg": tgt_diag,
                    "area_seg": tgt_area
                },
                "segmentation": {
                    "seg_path": seg_path,
                    "W": int(W_seg),
                    "H": int(H_seg)
                },
                "scanpath": scanpath,
                "summary": {
                    "success": {
                        "any": success_any,
                        "dwell": success_dwell,
                        "lastK": success_lastK,
                        "lastT": success_lastT
                    },
                    "found": {
                        "rule": f"dwell(min_hits={DWELL_MIN_HITS}) @ {rk_primary}",
                        "idx_found": None if idx_found is None else int(idx_found),
                        "t_found": None if t_found is None else float(t_found),
                        "time_to_target": None if time_to_target is None else float(time_to_target)
                    }
                }
            }

            f_out.write(json.dumps(trial, ensure_ascii=False) + "\n")
            nfix_list.append(len(scanpath))

            # Debug render
            if debug_done < DEBUG_N and os.path.exists(img_path):
                title = f"{pid}-{media_id}  {img_name}  n_fix={len(scanpath)}"
                out_path = os.path.join(DEBUG_DIR, f"debug_{debug_done:02d}_{img_stem}.png")
                render_debug(img_path, boxes, fix_pts_for_debug, debug_flag_for_points, tgt_box_seg,
                             out_path, title, W_seg, H_seg)
                debug_done += 1

    # Stats
    stats["trials_kept"] = int(len(nfix_list))
    stats["rows_clipped_to_01"] = int(clipped_count)

    if nfix_list:
        s = pd.Series(nfix_list).describe()
        stats["n_fix_per_trial_summary"] = {k: float(s[k]) for k in s.index}

    stats["element_class_counts"] = dict(elem_counter.most_common(30))

    if total_fix > 0:
        stats["background_ratio"] = float(total_bg / total_fix)
        stats["hit_ratio_non_background"] = float(total_hit / total_fix)

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_STATS)
    print("Debug images in:", DEBUG_DIR)
    print("Trials raw:", stats["trials_raw"])
    print("Trials kept:", stats["trials_kept"])
    print("Dropped (too short):", stats["trials_dropped_too_short"])
    print("Dropped (missing seg):", stats["trials_dropped_missing_seg"])
    print("Fix total:", total_fix)
    print("Background ratio:", round(stats["background_ratio"] or 0.0, 4))
    print("Hit ratio (non-background):", round(stats["hit_ratio_non_background"] or 0.0, 4))
    print("Clipped points:", clipped_count)


if __name__ == "__main__":
    main()
