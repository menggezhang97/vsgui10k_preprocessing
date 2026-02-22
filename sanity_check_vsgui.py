import os
import json
import glob
import gzip
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd

# optional viz (only when --viz enabled)
try:
    from PIL import Image
    import matplotlib.pyplot as plt
except Exception:
    Image = None
    plt = None


# ------------------------
# Utilities
# ------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_seg_index(seg_root: str):
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

def load_seg_shape(seg_path):
    d = load_json(seg_path)
    H, W, _ = d["img"]["shape"]
    return int(W), int(H)

def to_px(v, full):
    v = float(v)
    return v * float(full) if 0.0 <= v <= 1.5 else v

def map_px_to_seg(x_px, y_px, orig_W, orig_H, W_seg, H_seg):
    return float(x_px) * (W_seg / float(orig_W)), float(y_px) * (H_seg / float(orig_H))

def point_in_bbox(x, y, x0, y0, x1, y1):
    return (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)

def expand_bbox(x0, y0, x1, y1, r, W, H):
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = (x1 - x0) * r, (y1 - y0) * r
    nx0 = clamp(cx - 0.5 * w, 0, W); nx1 = clamp(cx + 0.5 * w, 0, W)
    ny0 = clamp(cy - 0.5 * h, 0, H); ny1 = clamp(cy + 0.5 * h, 0, H)
    return nx0, ny0, nx1, ny1

def stats_num(arr):
    if len(arr) == 0:
        return {}
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


# ------------------------
# Sanity checks
# ------------------------
def check_join_fix_targets(fix_csv, tgt_csv):
    fix = pd.read_csv(fix_csv, low_memory=False)
    tgt = pd.read_csv(tgt_csv, low_memory=False)

    fix_pairs = fix[["img_name", "tgt_id"]].drop_duplicates()
    if "ueyes_img_name" in tgt.columns:
        tgt_pairs = tgt[["ueyes_img_name", "tgt_id"]].drop_duplicates()
        joined = fix_pairs.merge(
            tgt_pairs,
            left_on=["img_name", "tgt_id"],
            right_on=["ueyes_img_name", "tgt_id"],
            how="inner",
        )
        note = "Joined fix(img_name,tgt_id) with targets(ueyes_img_name,tgt_id)."
    else:
        tgt_pairs = tgt[["img_name", "tgt_id"]].drop_duplicates()
        joined = fix_pairs.merge(tgt_pairs, on=["img_name", "tgt_id"], how="inner")
        note = "Joined fix(img_name,tgt_id) with targets(img_name,tgt_id)."

    return {
        "fix_unique_pairs": int(len(fix_pairs)),
        "tgt_unique_pairs": int(len(tgt_pairs)),
        "joined_pairs": int(len(joined)),
        "join_rate_vs_fix_pairs": float(len(joined) / max(1, len(fix_pairs))),
        "note": note,
    }

def check_cross_center(fix_csv, img_dir, seg_root, seed=7, sample_trials=600, min_dur=0.08):
    random.seed(seed)
    seg_idx = build_seg_index(seg_root)

    df = pd.read_csv(fix_csv, low_memory=False)
    df = df[(df["FPOGV"] == 1) & (df["BPOGV"] == 1) & (df["FPOGD"] >= min_dur)]
    df = df[df["img_type"] == 1]  # cross

    trial_key = ["pid", "img_name", "tgt_id", "cue", "absent"]
    groups = list(df.groupby(trial_key, sort=False).groups.keys())
    random.shuffle(groups)
    groups = groups[:min(sample_trials, len(groups))]

    dists = []
    used_trials = 0
    skipped_missing = 0

    gby = df.groupby(trial_key, sort=False)
    for key in groups:
        g = gby.get_group(key)
        r0 = g.iloc[0]
        img_name = str(r0["img_name"])
        stem = os.path.splitext(img_name)[0]

        img_path = os.path.join(img_dir, img_name)
        seg_path = seg_idx.get(stem)
        if (not os.path.exists(img_path)) or (not seg_path) or (not os.path.exists(seg_path)):
            skipped_missing += 1
            continue

        W_seg, H_seg = load_seg_shape(seg_path)
        orig_W = float(r0["original_width"])
        orig_H = float(r0["original_height"])
        cx, cy = 0.5 * W_seg, 0.5 * H_seg

        for _, row in g.iterrows():
            raw_fx = float(row["FPOGX_scaled"])
            raw_fy = float(row["FPOGY_scaled"])
            fx = to_px(raw_fx, orig_W)
            fy = to_px(raw_fy, orig_H)
            sx, sy = map_px_to_seg(fx, fy, orig_W, orig_H, W_seg, H_seg)
            sx = clamp(sx, 0, W_seg)
            sy = clamp(sy, 0, H_seg)
            dists.append(float(np.hypot(sx - cx, sy - cy)))

        used_trials += 1

    return {
        "used_trials": int(used_trials),
        "skipped_missing_img_or_seg": int(skipped_missing),
        "dist_to_center_px": stats_num(dists),
        "interpretation": "Lower is better. If mapping is correct, cross fixations should cluster near center.",
    }

def read_trials_jsonl(jsonl_gz_path, max_trials=None):
    trials = []
    with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_trials is not None and i >= max_trials:
                break
            trials.append(json.loads(line))
    return trials

def check_trial_times(trials, n_sample=200, seed=7):
    random.seed(seed)
    sample = trials if len(trials) <= n_sample else random.sample(trials, n_sample)

    bad = []
    for t in sample:
        tid = t.get("trial_id", "NA")
        s = t["phases"]["search"]["scanpath"]
        ts = [p["t"] for p in s]
        if len(ts) == 0:
            bad.append({"trial_id": tid, "issue": "empty_search"})
            continue
        if min(ts) < -1e-6:
            bad.append({"trial_id": tid, "issue": "negative_search_t"})
        if any(ts[i] > ts[i+1] + 1e-6 for i in range(len(ts)-1)):
            bad.append({"trial_id": tid, "issue": "search_t_not_monotonic"})
        t_end = t["phases"]["search"]["t_end"]
        if abs(float(ts[-1]) - float(t_end)) > 2.0:
            bad.append({"trial_id": tid, "issue": "search_t_end_mismatch"})

        v = t["phases"]["validation"]
        if v.get("available", 0) == 1 and v.get("n_fix", 0) > 0:
            tv = [p["t"] for p in v["fixations"]]
            if min(tv) < -1e-6:
                bad.append({"trial_id": tid, "issue": "negative_valid_t"})
            if any(tv[i] > tv[i+1] + 1e-6 for i in range(len(tv)-1)):
                bad.append({"trial_id": tid, "issue": "valid_t_not_monotonic"})

    return {
        "checked_trials": int(len(sample)),
        "n_issues": int(len(bad)),
        "issues_examples": bad[:20],
        "interpretation": "Search/validation relative times should start at ~0 and be non-decreasing.",
    }

def check_search_hit_rate_from_trials(trials, r_list=(1.0, 1.5, 2.0, 3.0)):
    agg = {r: {"n": 0, "any": 0, "dwell_min2": 0} for r in r_list}
    skipped = 0

    for t in trials:
        absent = int(t["key"]["absent"])
        if absent == 1:
            continue
        bbox = t["target"].get("bbox_seg", None)
        if bbox is None:
            skipped += 1
            continue

        W = t["geom"]["seg_w"]
        H = t["geom"]["seg_h"]
        x0 = float(bbox["x0"]); y0 = float(bbox["y0"]); x1 = float(bbox["x1"]); y1 = float(bbox["y1"])
        pts = [(p["xy_seg"]["x"], p["xy_seg"]["y"]) for p in t["phases"]["search"]["scanpath"]]

        for r in r_list:
            ex0, ey0, ex1, ey1 = expand_bbox(x0, y0, x1, y1, float(r), W, H)
            n_inside = int(sum(point_in_bbox(px, py, ex0, ey0, ex1, ey1) for (px, py) in pts))
            agg[r]["n"] += 1
            agg[r]["any"] += int(n_inside >= 1)
            agg[r]["dwell_min2"] += int(n_inside >= 2)

    out = {}
    for r, d in agg.items():
        n = max(1, d["n"])
        out[str(r)] = {
            "n_trials": int(d["n"]),
            "any_rate": float(d["any"] / n),
            "dwell_min2_rate": float(d["dwell_min2"] / n),
        }
    return {
        "present_only_search_hit_rates": out,
        "skipped_missing_bbox_seg": int(skipped),
        "interpretation": "Rates should increase with r. Very low rates for all r indicates bbox/coord mismatch.",
    }

def check_phase_durations(trials):
    search_d = [t["phases"]["search"]["t_end"] for t in trials]
    valid_d = [t["phases"]["validation"]["t_end"] for t in trials if t["phases"]["validation"]["t_end"] is not None]
    return {
        "search_duration_s": stats_num(search_d),
        "validation_duration_s": stats_num(valid_d),
        "interpretation": "Search median often ~seconds; validation often ~3s (but depends on dataset).",
    }

def check_ui_mapping(trials):
    """
    Includes:
      - class distribution
      - none rate
      - overlap distribution
      - top hit elements
      - NEW: mode distribution (inside/nearest/none)
    """
    def collect(phase_name):
        cls = Counter()
        overlap = Counter()
        mode = Counter()
        none_cnt = 0
        total = 0
        top_elem = Counter()

        for t in trials:
            img = t["key"]["img_name"]
            if phase_name == "search":
                items = t["phases"]["search"]["scanpath"]
            else:
                v = t["phases"]["validation"]
                items = v["fixations"] if v.get("available", 0) == 1 else []

            for p in items:
                ui = p.get("ui", None) or {}

                c = ui.get("class", None)
                ov = int(ui.get("overlap", 0) or 0)
                idx = ui.get("idx", None)
                m = ui.get("mode", None)  # expected: inside / nearest / none

                total += 1
                overlap[ov] += 1

                # mode
                if m is None:
                    # backward compat: infer
                    m = "none" if c is None else "inside"
                mode[str(m)] += 1

                # class
                if c is None:
                    none_cnt += 1
                    cls["None"] += 1
                else:
                    cls[str(c)] += 1
                    if idx is not None:
                        top_elem[(img, int(idx), str(c))] += 1

        cls_rate = {k: float(v) / max(1, total) for k, v in cls.items()}
        mode_rate = {k: float(v) / max(1, total) for k, v in mode.items()}
        none_rate = float(none_cnt) / max(1, total)

        overlap_top = dict(overlap.most_common(10))

        top10 = []
        for (img, idx, c), v in top_elem.most_common(10):
            top10.append({"img_name": img, "ui_idx": idx, "class": c, "n_fix": int(v)})

        return {
            "n_fixations": int(total),
            "class_counts": dict(cls),
            "class_rates": cls_rate,
            "none_rate": none_rate,
            "mode_counts": dict(mode),
            "mode_rates": mode_rate,
            "overlap_top": overlap_top,
            "top_hit_elements": top10,
        }

    return {
        "search": collect("search"),
        "validation": collect("validation"),
        "interpretation": (
            "None_rate should be relatively low. Mode 'inside' should dominate; some 'nearest' is expected."
        ),
    }

def check_file_coverage(trials, img_dir, seg_root):
    seg_idx = build_seg_index(seg_root)
    miss_img = 0
    miss_seg = 0
    for t in trials:
        img_name = t["key"]["img_name"]
        stem = os.path.splitext(img_name)[0]
        if not os.path.exists(os.path.join(img_dir, img_name)):
            miss_img += 1
        seg_path = seg_idx.get(stem)
        if (seg_path is None) or (not os.path.exists(seg_path)):
            miss_seg += 1
    n = len(trials)
    return {
        "n_trials_checked": int(n),
        "missing_image_trials": int(miss_img),
        "missing_seg_trials": int(miss_seg),
        "missing_image_rate": float(miss_img / max(1, n)),
        "missing_seg_rate": float(miss_seg / max(1, n)),
        "interpretation": "Should be near 0 if dataset paths are consistent with preprocessing assumptions.",
    }


# ------------------------
# None-only visualization (optional)
# ------------------------
def _can_viz():
    return (Image is not None) and (plt is not None)

def viz_none_fixations(trials, img_dir, seg_root, out_dir, seed=7, n_viz=8, phase="search"):
    """
    Draw ONLY fixations whose ui.class is None (or ui.mode=='none'), to see where None comes from.
    Uses seg_w/seg_h from trials to match your mapping space.
    """
    if not _can_viz():
        return {"enabled": 0, "reason": "PIL/matplotlib not available", "saved": []}

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    seg_idx = build_seg_index(seg_root)

    cand = []
    for t in trials:
        img_name = t["key"]["img_name"]
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        seg_path = seg_idx.get(stem)
        if (not os.path.exists(img_path)) or (seg_path is None) or (not os.path.exists(seg_path)):
            continue

        if phase == "search":
            items = t["phases"]["search"]["scanpath"]
        else:
            v = t["phases"]["validation"]
            items = v["fixations"] if v.get("available", 0) == 1 else []

        none_pts = []
        for p in items:
            ui = p.get("ui", None) or {}
            c = ui.get("class", None)
            m = ui.get("mode", None)
            if (c is None) or (m == "none"):
                xy = p.get("xy_seg", {})
                none_pts.append((float(xy.get("x", 0.0)), float(xy.get("y", 0.0))))

        if len(none_pts) > 0:
            cand.append((t, img_path, none_pts))

    if not cand:
        return {"enabled": 1, "reason": "no trials have none fixations", "saved": []}

    random.shuffle(cand)
    cand = cand[:min(n_viz, len(cand))]

    saved = []
    for i, (t, img_path, pts) in enumerate(cand):
        W = int(t["geom"]["seg_w"])
        H = int(t["geom"]["seg_h"])

        img = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"NONE-only {phase.upper()} | {t['trial_id']} | n_none={len(pts)}")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.scatter(xs, ys, s=55, marker="o", c="yellow", edgecolors="black", linewidths=0.7)
        plt.xlim([0, W])
        plt.ylim([H, 0])
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"none_{phase}_{i:02d}_{os.path.splitext(t['key']['img_name'])[0]}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        saved.append(out_path)

    return {"enabled": 1, "reason": "ok", "saved": saved}


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--trials_jsonl", type=str, default="trials_official_with_validation.jsonl.gz")
    ap.add_argument("--out_report", type=str, default="sanity_report.json")
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)

    # NEW optional viz
    ap.add_argument("--viz_none", action="store_true", help="Save none-only fixation overlays")
    ap.add_argument("--viz_out", type=str, default="sanity_viz_none", help="Output dir for viz")
    ap.add_argument("--viz_n", type=int, default=8, help="How many none-only images to save per phase")
    args = ap.parse_args()

    data_root = args.data_root
    fix_csv = os.path.join(data_root, "vsgui10k_fixations.csv")
    tgt_csv = os.path.join(data_root, "vsgui10k_targets.csv")
    img_dir = os.path.join(data_root, "vsgui10k-images")
    seg_root = os.path.join(data_root, "segmentation")
    trials_path = args.trials_jsonl

    assert os.path.exists(fix_csv), f"Missing: {fix_csv}"
    assert os.path.exists(tgt_csv), f"Missing: {tgt_csv}"
    assert os.path.isdir(img_dir), f"Missing: {img_dir}"
    assert os.path.isdir(seg_root), f"Missing: {seg_root}"
    assert os.path.exists(trials_path), f"Missing: {trials_path}"

    trials = read_trials_jsonl(trials_path, max_trials=args.max_trials)
    print(f"Loaded trials: {len(trials)} from {trials_path}")

    report = {
        "paths": {
            "data_root": data_root,
            "fix_csv": fix_csv,
            "tgt_csv": tgt_csv,
            "img_dir": img_dir,
            "seg_root": seg_root,
            "trials_jsonl": trials_path,
        },
        "checks": {}
    }

    report["checks"]["file_coverage_from_trials"] = check_file_coverage(trials, img_dir, seg_root)
    report["checks"]["join_fixations_targets"] = check_join_fix_targets(fix_csv, tgt_csv)
    report["checks"]["cross_center_dispersion"] = check_cross_center(fix_csv, img_dir, seg_root, seed=args.seed)
    report["checks"]["trial_time_monotonicity"] = check_trial_times(trials, seed=args.seed)
    report["checks"]["phase_durations"] = check_phase_durations(trials)
    report["checks"]["search_hit_rate_from_trials"] = check_search_hit_rate_from_trials(trials)
    report["checks"]["ui_mapping"] = check_ui_mapping(trials)

    # Optional none-only viz
    if args.viz_none:
        out_dir = args.viz_out
        report["checks"]["viz_none_search"] = viz_none_fixations(
            trials, img_dir, seg_root, os.path.join(out_dir, "search"),
            seed=args.seed, n_viz=args.viz_n, phase="search"
        )
        report["checks"]["viz_none_validation"] = viz_none_fixations(
            trials, img_dir, seg_root, os.path.join(out_dir, "validation"),
            seed=args.seed, n_viz=args.viz_n, phase="validation"
        )

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print concise summary
    print("\n=== SANITY SUMMARY ===")
    j = report["checks"]["join_fixations_targets"]
    print(f"[JOIN] join_rate_vs_fix_pairs = {j['join_rate_vs_fix_pairs']:.4f} ({j['joined_pairs']}/{j['fix_unique_pairs']})")

    c = report["checks"]["cross_center_dispersion"]["dist_to_center_px"]
    if c:
        print(f"[CROSS] mean_dist_to_center_px = {c.get('mean', None):.2f}, p50={c.get('p50', None):.2f}, p90={c.get('p90', None):.2f}")

    tchk = report["checks"]["trial_time_monotonicity"]
    print(f"[TIME] checked={tchk['checked_trials']} issues={tchk['n_issues']}")

    durs = report["checks"]["phase_durations"]
    print(f"[DUR] search_p50={durs['search_duration_s'].get('p50', None):.2f}s  valid_p50={durs['validation_duration_s'].get('p50', None):.2f}s")

    hr = report["checks"]["search_hit_rate_from_trials"]["present_only_search_hit_rates"]
    print(f"[HIT] present-only any_rate(r=1,2,3) = {hr.get('1.0',{}).get('any_rate',0):.3f}, {hr.get('2.0',{}).get('any_rate',0):.3f}, {hr.get('3.0',{}).get('any_rate',0):.3f}")

    ui_s = report["checks"]["ui_mapping"]["search"]
    top_s = sorted(ui_s["class_rates"].items(), key=lambda x: -x[1])[:4]
    print(f"[UI-SEARCH] none_rate={ui_s['none_rate']:.3f} mode_rates={ui_s.get('mode_rates',{})} top_classes={top_s}")

    ui_v = report["checks"]["ui_mapping"]["validation"]
    top_v = sorted(ui_v["class_rates"].items(), key=lambda x: -x[1])[:4]
    print(f"[UI-VALID]  none_rate={ui_v['none_rate']:.3f} mode_rates={ui_v.get('mode_rates',{})} top_classes={top_v}")

    cov = report["checks"]["file_coverage_from_trials"]
    print(f"[COV] missing_img_rate={cov['missing_image_rate']:.3f} missing_seg_rate={cov['missing_seg_rate']:.3f}")

    if args.viz_none:
        s_saved = report["checks"].get("viz_none_search", {}).get("saved", [])
        v_saved = report["checks"].get("viz_none_validation", {}).get("saved", [])
        print(f"[VIZ] none-only saved search={len(s_saved)} valid={len(v_saved)} -> {args.viz_out}")

    print(f"\nSaved report -> {args.out_report}")


if __name__ == "__main__":
    main()