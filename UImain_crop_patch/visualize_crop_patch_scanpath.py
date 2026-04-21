import os
import json
import gzip
import random
import argparse
import sys
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

# =========================================================
# Force local module resolution first
# =========================================================
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# =========================================================
# Import the REAL training-time model
# =========================================================
from model_crop_decoder import HybridPlainDecoderModel

print("HybridPlainDecoderModel imported from:", inspect.getfile(HybridPlainDecoderModel))
print("Forward args:", inspect.signature(HybridPlainDecoderModel.forward))


# =========================================================
# 1. Utils
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trials_jsonl_gz(path):
    trials = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def split_trials(trials, train_ratio, val_ratio, test_ratio, seed):
    idx = list(range(len(trials)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_trials = [trials[i] for i in train_idx]
    val_trials = [trials[i] for i in val_idx]
    test_trials = [trials[i] for i in test_idx]
    return train_trials, val_trials, test_trials


def normalize_rgb(arr):
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def build_image_tensor(image_path, image_size):
    img = Image.open(image_path).convert("RGB")
    raw_w, raw_h = img.size
    img_resized = img.resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(img_resized)
    arr = normalize_rgb(arr)
    return torch.tensor(arr, dtype=torch.float32), img, raw_w, raw_h


def resolve_image_path(trial, image_dir):
    img_name = trial["key"]["img_name"]
    p = Path(image_dir) / img_name
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return str(p)


def resolve_seg_path(trial, seg_root):
    img_name = trial["key"]["img_name"]
    stem = Path(img_name).stem

    patterns = [
        f"{stem}.json",
        f"{img_name}.json",
        f"{stem}.png.json",
    ]

    seg_root = Path(seg_root)

    for pat in patterns:
        p = seg_root / pat
        if p.exists():
            return str(p)

    for pat in patterns:
        matches = list(seg_root.rglob(pat))
        if len(matches) == 1:
            return str(matches[0])
        elif len(matches) > 1:
            print(f"[resolve_seg_path] multiple matches for {pat}, using first:")
            for m in matches[:10]:
                print("  ", m)
            return str(matches[0])

    raise FileNotFoundError(
        f"Seg file not found for img_name={img_name} under seg_root={seg_root}"
    )


def extract_cue_text(trial):
    return str(trial["key"]["cue"]).strip()


def extract_gt_fixations(trial):
    scanpath = trial["phases"]["search"]["scanpath"]
    out = []
    for item in scanpath:
        x = float(item["xy_img_raw_scaled"]["x"])
        y = float(item["xy_img_raw_scaled"]["y"])
        d = float(item.get("dur", 0.0))
        out.append([x, y, d])
    if len(out) < 2:
        raise ValueError("Need at least 2 fixations.")
    return out


def parse_segmentation(seg_json_path, seg_w=None, seg_h=None):
    seg = load_json(seg_json_path)

    if seg_w is not None and seg_h is not None:
        width = float(seg_w)
        height = float(seg_h)
    else:
        if "img" in seg and "shape" in seg["img"]:
            shape = seg["img"]["shape"]
            if len(shape) >= 2:
                height = float(shape[0])
                width = float(shape[1])
            else:
                raise KeyError(f"Invalid img.shape in {seg_json_path}")
        else:
            raise KeyError(f"Cannot find seg width/height in {seg_json_path}")

    if "compos" in seg and isinstance(seg["compos"], list):
        elements = seg["compos"]
    else:
        raise KeyError(f"Cannot find compos list in {seg_json_path}")

    return width, height, elements


def parse_bbox(elem):
    if all(k in elem for k in ["column_min", "column_max", "row_min", "row_max"]):
        x0 = float(elem["column_min"])
        x1 = float(elem["column_max"])
        y0 = float(elem["row_min"])
        y1 = float(elem["row_max"])
        return x0, y0, x1 - x0, y1 - y0

    if "bbox" in elem:
        b = elem["bbox"]
    elif "bounds" in elem:
        b = elem["bounds"]
    elif "rect" in elem:
        b = elem["rect"]
    elif "box" in elem:
        b = elem["box"]
    else:
        return None

    if isinstance(b, (list, tuple)) and len(b) == 4:
        x, y, w, h = map(float, b)
        return x, y, w, h

    if isinstance(b, dict):
        if all(k in b for k in ["x", "y", "w", "h"]):
            return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        if all(k in b for k in ["x0", "y0", "x1", "y1"]):
            x0, y0, x1, y1 = map(float, [b["x0"], b["y0"], b["x1"], b["y1"]])
            return x0, y0, x1 - x0, y1 - y0
        if all(k in b for k in ["left", "top", "width", "height"]):
            return float(b["left"]), float(b["top"]), float(b["width"]), float(b["height"])

    return None


def parse_type(elem):
    for key in ["class", "type", "label", "category", "name"]:
        if key in elem:
            return str(elem[key])
    return "Background"


def clamp_bbox_to_image(x, y, w, h, img_w, img_h):
    """
    Return a safe crop box (x0, y0, x1, y1) in PIL convention.
    Guarantees:
      0 <= x0 < x1 <= img_w
      0 <= y0 < y1 <= img_h
    """
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(w) or not np.isfinite(h):
        return 0, 0, 1, 1

    if w <= 0:
        w = 1.0
    if h <= 0:
        h = 1.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))

    # x dimension
    if x0 >= img_w:
        x0 = img_w - 1
        x1 = img_w
    elif x1 <= 0:
        x0 = 0
        x1 = 1
    else:
        x0 = max(0, x0)
        x1 = min(img_w, x1)
        if x1 <= x0:
            x1 = min(img_w, x0 + 1)
            if x1 <= x0:
                x0 = max(0, img_w - 1)
                x1 = img_w

    # y dimension
    if y0 >= img_h:
        y0 = img_h - 1
        y1 = img_h
    elif y1 <= 0:
        y0 = 0
        y1 = 1
    else:
        y0 = max(0, y0)
        y1 = min(img_h, y1)
        if y1 <= y0:
            y1 = min(img_h, y0 + 1)
            if y1 <= y0:
                y0 = max(0, img_h - 1)
                y1 = img_h

    return x0, y0, x1, y1


def build_ui_crop_tensor(raw_img, bbox, crop_size):
    """
    raw_img: PIL.Image in original resolution
    bbox: (x, y, w, h) in raw image pixel space
    return: torch.FloatTensor [3, crop_size, crop_size]
    """
    img_w, img_h = raw_img.size
    x, y, w, h = bbox
    x0, y0, x1, y1 = clamp_bbox_to_image(x, y, w, h, img_w, img_h)

    if x1 <= x0 or y1 <= y0:
        fallback = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        arr = normalize_rgb(fallback)
        return torch.tensor(arr, dtype=torch.float32)

    try:
        crop = raw_img.crop((x0, y0, x1, y1)).convert("RGB")
    except Exception:
        fallback = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        arr = normalize_rgb(fallback)
        return torch.tensor(arr, dtype=torch.float32)

    crop = crop.resize((crop_size, crop_size), Image.BILINEAR)

    arr = np.asarray(crop)
    arr = normalize_rgb(arr)
    return torch.tensor(arr, dtype=torch.float32)


def load_ui_tokens(
    seg_json_path,
    ui_type_vocab,
    max_ui_tokens,
    drop_full_screen_root,
    raw_img,
    crop_size,
    seg_w=None,
    seg_h=None,
    debug_bbox=False,
):
    """
    return:
      ui_geom: [Ku, 4]
      ui_type_id: [Ku]
      ui_mask: [Ku] bool
      ui_crop_images: [Ku, 3, Hc, Wc]
    """
    seg_w, seg_h, elements = parse_segmentation(seg_json_path, seg_w=seg_w, seg_h=seg_h)

    raw_w, raw_h = raw_img.size

    ui_geom = []
    ui_type_id = []
    ui_mask = []
    ui_crop_images = []

    def is_full_root(x, y, w, h):
        return abs(x) < 1e-6 and abs(y) < 1e-6 and abs(w - seg_w) < 1e-6 and abs(h - seg_h) < 1e-6

    for elem in elements:
        bbox = parse_bbox(elem)
        if bbox is None:
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue

        cls_name = parse_type(elem)
        if cls_name not in ui_type_vocab:
            cls_name = "<UNK>"

        if drop_full_screen_root and cls_name in ["Background", "root", "Root"]:
            if is_full_root(x, y, w, h):
                continue

        # normalized geometry in segmentation coordinate space
        xc = (x + 0.5 * w) / float(seg_w)
        yc = (y + 0.5 * h) / float(seg_h)
        wn = w / float(seg_w)
        hn = h / float(seg_h)

        # map bbox from segmentation coordinates to raw image coordinates
        x_img = x / float(seg_w) * raw_w
        y_img = y / float(seg_h) * raw_h
        w_img = w / float(seg_w) * raw_w
        h_img = h / float(seg_h) * raw_h

        if debug_bbox and len(ui_geom) < 5:
            print(
                "[debug bbox]",
                "seg_bbox=", (x, y, w, h),
                "img_bbox=", (x_img, y_img, w_img, h_img),
                "raw_size=", (raw_w, raw_h),
                "seg_size=", (seg_w, seg_h),
            )

        crop_t = build_ui_crop_tensor(raw_img, (x_img, y_img, w_img, h_img), crop_size)

        ui_geom.append([xc, yc, wn, hn])
        ui_type_id.append(ui_type_vocab.get(cls_name, ui_type_vocab.get("<UNK>", 1)))
        ui_mask.append(1)
        ui_crop_images.append(crop_t)

        if len(ui_geom) >= max_ui_tokens:
            break

    pad_id = ui_type_vocab.get("<PAD>", 0)
    while len(ui_geom) < max_ui_tokens:
        ui_geom.append([0.0, 0.0, 0.0, 0.0])
        ui_type_id.append(pad_id)
        ui_mask.append(0)
        ui_crop_images.append(torch.zeros(3, crop_size, crop_size, dtype=torch.float32))

    return (
        torch.tensor(ui_geom, dtype=torch.float32),
        torch.tensor(ui_type_id, dtype=torch.long),
        torch.tensor(ui_mask, dtype=torch.bool),
        torch.stack(ui_crop_images, dim=0),
    )


def to_pixel_path(path_xy, w, h):
    arr = np.array(path_xy, dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
    arr[:, 0] *= w
    arr[:, 1] *= h
    return arr


@torch.no_grad()
def predict_next(
    model,
    device,
    image_t,
    cue_id_t,
    ui_geom_t,
    ui_type_id_t,
    ui_mask_t,
    ui_crop_images_t,
    history_seq,
    history_len,
):
    hist = history_seq[-history_len:]
    hist_t = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(device)

    pred = model(
        image=image_t.unsqueeze(0).to(device),
        cue_id=cue_id_t.unsqueeze(0).to(device),
        history_xydur=hist_t,
        ui_geom=ui_geom_t.unsqueeze(0).to(device),
        ui_type_id=ui_type_id_t.unsqueeze(0).to(device),
        ui_mask=ui_mask_t.unsqueeze(0).to(device),
        ui_crop_images=ui_crop_images_t.unsqueeze(0).to(device),
    )
    pred_xy = pred[0].detach().cpu().numpy().tolist()
    return pred_xy


def teacher_forced_sequence(
    model,
    device,
    image_t,
    cue_id_t,
    ui_geom_t,
    ui_type_id_t,
    ui_mask_t,
    ui_crop_images_t,
    gt_fixations,
    history_len,
):
    """
    Teacher-forced:
    next prediction conditioned on GT history
    """
    gt_seq = [[p[0], p[1]] for p in gt_fixations]
    pred_seq = [gt_seq[0]]

    T = len(gt_fixations)
    for t in range(1, T):
        hist = gt_fixations[max(0, t - history_len):t]
        pred_xy = predict_next(
            model=model,
            device=device,
            image_t=image_t,
            cue_id_t=cue_id_t,
            ui_geom_t=ui_geom_t,
            ui_type_id_t=ui_type_id_t,
            ui_mask_t=ui_mask_t,
            ui_crop_images_t=ui_crop_images_t,
            history_seq=hist,
            history_len=history_len,
        )
        pred_seq.append(pred_xy)

    return gt_seq, pred_seq


def rollout_sequence(
    model,
    device,
    image_t,
    cue_id_t,
    ui_geom_t,
    ui_type_id_t,
    ui_mask_t,
    ui_crop_images_t,
    gt_fixations,
    history_len,
    rollout_steps=None,
):
    """
    Rollout:
    first fixation = GT first fixation
    then autoregressive predictions fed back as history
    """
    gt_seq = [[p[0], p[1]] for p in gt_fixations]
    if rollout_steps is None:
        rollout_steps = len(gt_fixations)

    pred_hist = [gt_fixations[0][:]]
    pred_seq = [gt_seq[0]]

    for _ in range(1, rollout_steps):
        pred_xy = predict_next(
            model=model,
            device=device,
            image_t=image_t,
            cue_id_t=cue_id_t,
            ui_geom_t=ui_geom_t,
            ui_type_id_t=ui_type_id_t,
            ui_mask_t=ui_mask_t,
            ui_crop_images_t=ui_crop_images_t,
            history_seq=pred_hist,
            history_len=history_len,
        )
        pred_seq.append(pred_xy)
        pred_hist.append([pred_xy[0], pred_xy[1], 0.0])

    return gt_seq[:rollout_steps], pred_seq


def draw_overlay(pil_img, gt_path, pred_path, out_path, title, target_bbox=None):
    img = np.array(pil_img)
    h, w = img.shape[:2]

    gt_px = to_pixel_path(gt_path, w, h)
    pred_px = to_pixel_path(pred_path, w, h)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(title)

    plt.plot(
        gt_px[:, 0], gt_px[:, 1],
        "-o",
        color="deepskyblue",
        linewidth=2,
        markersize=4,
        label="GT"
    )
    plt.plot(
        pred_px[:, 0], pred_px[:, 1],
        "-o",
        color="orangered",
        linewidth=2,
        markersize=4,
        label="Pred"
    )

    for i, (x, y) in enumerate(gt_px):
        plt.text(x + 3, y + 3, f"G{i+1}", color="deepskyblue", fontsize=7)
    for i, (x, y) in enumerate(pred_px):
        plt.text(x + 3, y - 6, f"P{i+1}", color="orangered", fontsize=7)

    if target_bbox is not None:
        tx = target_bbox["x"] * w
        ty = target_bbox["y"] * h
        tw = target_bbox["w"] * w
        th = target_bbox["h"] * h
        rect = plt.Rectangle(
            (tx, ty),
            tw,
            th,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            linestyle="--",
            label="Target"
        )
        plt.gca().add_patch(rect)

    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. Batch visualization helpers
# =========================================================

def pick_indices(split_trials_list, seed, num_samples):
    rng = random.Random(seed)
    n = len(split_trials_list)
    k = min(num_samples, n)
    return rng.sample(range(n), k=k)


def process_one_trial(
    picked_idx,
    split_name,
    trial,
    model,
    device,
    cfg,
    cue_vocab,
    ui_type_vocab,
    image_dir,
    seg_root,
    output_dir,
    checkpoint_path,
    checkpoint_epoch,
    checkpoint_best_loss,
    checkpoint_best_dist,
    crop_size,
    rollout_steps=None,
    debug_bbox=False,
):
    print(f"\n[{split_name}] picked trial index: {picked_idx}")
    print("trial_id:", trial["trial_id"])

    split_out_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    image_path = resolve_image_path(trial, image_dir)
    seg_json_path = resolve_seg_path(trial, seg_root)
    cue_text = extract_cue_text(trial)

    image_t, raw_img, raw_w, raw_h = build_image_tensor(image_path, cfg["image_size"])
    gt_fix = extract_gt_fixations(trial)

    cue_key = cue_text if cue_text in cue_vocab else cue_text.lower()
    cue_id = cue_vocab.get(cue_key, cue_vocab.get("<UNK>", 0))
    cue_id_t = torch.tensor(cue_id, dtype=torch.long)

    print("cue_text:", cue_text, "-> cue_id:", cue_id)

    trial_seg_w = trial["geom"]["seg_w"]
    trial_seg_h = trial["geom"]["seg_h"]

    ui_geom_t, ui_type_id_t, ui_mask_t, ui_crop_images_t = load_ui_tokens(
        seg_json_path=seg_json_path,
        ui_type_vocab=ui_type_vocab,
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
        raw_img=raw_img,
        crop_size=crop_size,
        seg_w=trial_seg_w,
        seg_h=trial_seg_h,
        debug_bbox=debug_bbox,
    )

    target_bbox = trial.get("target", {}).get("bbox_img_raw", None)

    # Teacher-forced
    gt_tf, pred_tf = teacher_forced_sequence(
        model=model,
        device=device,
        image_t=image_t,
        cue_id_t=cue_id_t,
        ui_geom_t=ui_geom_t,
        ui_type_id_t=ui_type_id_t,
        ui_mask_t=ui_mask_t,
        ui_crop_images_t=ui_crop_images_t,
        gt_fixations=gt_fix,
        history_len=cfg["history_len"],
    )

    tf_png = os.path.join(split_out_dir, f"trial_{picked_idx:05d}_teacher_forced.png")
    draw_overlay(
        raw_img,
        gt_tf,
        pred_tf,
        tf_png,
        title=f"{split_name} | Teacher-forced | trial={picked_idx} | cue={cue_text}",
        target_bbox=target_bbox,
    )

    # Rollout
    gt_ro, pred_ro = rollout_sequence(
        model=model,
        device=device,
        image_t=image_t,
        cue_id_t=cue_id_t,
        ui_geom_t=ui_geom_t,
        ui_type_id_t=ui_type_id_t,
        ui_mask_t=ui_mask_t,
        ui_crop_images_t=ui_crop_images_t,
        gt_fixations=gt_fix,
        history_len=cfg["history_len"],
        rollout_steps=rollout_steps,
    )

    ro_png = os.path.join(split_out_dir, f"trial_{picked_idx:05d}_rollout.png")
    draw_overlay(
        raw_img,
        gt_ro,
        pred_ro,
        ro_png,
        title=f"{split_name} | Rollout | trial={picked_idx} | cue={cue_text}",
        target_bbox=target_bbox,
    )

    out_json = os.path.join(split_out_dir, f"trial_{picked_idx:05d}_viz.json")
    out_data = {
        "split": split_name,
        "trial_index": picked_idx,
        "trial_id": trial["trial_id"],
        "image_path": image_path,
        "seg_json_path": seg_json_path,
        "cue_text": cue_text,

        "model_variant": "ui_crop_decoder",
        "visual_mode": "ui_crop_matched",
        "ui_memory_inputs": ["geom", "type", "crop"],
        "crop_size": crop_size,

        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint_epoch,
        "best_val_loss": checkpoint_best_loss,
        "best_val_dist": checkpoint_best_dist,

        "teacher_forced_definition": {
            "history_source": "ground_truth",
            "prediction_target": "next_fixation_xy",
            "history_contains_duration": True,
        },
        "rollout_definition": {
            "history_source": "model_predictions",
            "prediction_target": "next_fixation_xy",
            "predicted_duration_fallback": 0.0,
        },

        "teacher_forced_gt": gt_tf,
        "teacher_forced_pred": pred_tf,
        "rollout_gt": gt_ro,
        "rollout_pred": pred_ro,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print("  ", tf_png)
    print("  ", ro_png)
    print("  ", out_json)

    return {
        "split": split_name,
        "trial_index": picked_idx,
        "trial_id": trial["trial_id"],
        "cue_text": cue_text,
        "teacher_forced_png": tf_png,
        "rollout_png": ro_png,
        "json_path": out_json,
    }


# =========================================================
# 4. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trials_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--seg_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="viz_outputs")
    parser.add_argument(
        "--trial_index",
        type=int,
        default=None,
        help="Optional: visualize only one test trial index."
    )
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument(
        "--num_samples_per_split",
        type=int,
        default=10,
        help="How many reproducible samples to draw from each split."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--ui_crop_size",
        type=int,
        required=True,
        help="Explicit UI crop resize size used by this crop model."
    )
    parser.add_argument(
        "--debug_bbox",
        action="store_true",
        help="Print first few bbox mappings for debugging."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    cue_vocab = ckpt["cue_vocab"]
    ui_type_vocab = ckpt["ui_type_vocab"]

    print("=" * 80)
    print("checkpoint:", args.checkpoint)
    print("epoch:", ckpt.get("epoch", None))
    print("best_val_loss:", ckpt.get("best_val_loss", None))
    print("best_val_dist:", ckpt.get("best_val_dist", None))
    print("ui_crop_size (explicit):", args.ui_crop_size)
    print("=" * 80)

    device = torch.device(args.device)

    model = HybridPlainDecoderModel(
        vit_name=cfg["vit_name"],
        pretrained=False,
        cue_vocab_size=len(cue_vocab),
        ui_type_vocab_size=len(ui_type_vocab),
        history_len=cfg["history_len"],
        ui_geom_dim=cfg["ui_geom_dim"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        ui_memory_scale=cfg.get("ui_memory_scale", 1.0),
        freeze_patch_backbone=cfg.get("freeze_patch_backbone", False),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    trials = load_trials_jsonl_gz(args.trials_path)
    train_trials, val_trials, test_trials = split_trials(
        trials,
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg["test_ratio"],
        cfg["seed"],
    )

    summary = {
        "seed": cfg["seed"],
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "best_val_dist": ckpt.get("best_val_dist", None),
        "num_samples_per_split": args.num_samples_per_split,
        "ui_crop_size": args.ui_crop_size,
        "picked_indices": {},
        "records": [],
    }

    if args.trial_index is not None:
        picked = {"test": [args.trial_index]}
    else:
        picked = {
            "train": pick_indices(train_trials, cfg["seed"] + 101, args.num_samples_per_split),
            "val": pick_indices(val_trials, cfg["seed"] + 202, args.num_samples_per_split),
            "test": pick_indices(test_trials, cfg["seed"] + 303, args.num_samples_per_split),
        }

    summary["picked_indices"] = picked

    picked_indices_path = os.path.join(args.output_dir, "picked_trial_indices.json")
    with open(picked_indices_path, "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)

    split_map = {
        "train": train_trials,
        "val": val_trials,
        "test": test_trials,
    }

    for split_name, indices in picked.items():
        for picked_idx in indices:
            trial = split_map[split_name][picked_idx]
            record = process_one_trial(
                picked_idx=picked_idx,
                split_name=split_name,
                trial=trial,
                model=model,
                device=device,
                cfg=cfg,
                cue_vocab=cue_vocab,
                ui_type_vocab=ui_type_vocab,
                image_dir=args.image_dir,
                seg_root=args.seg_root,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                checkpoint_epoch=ckpt.get("epoch", None),
                checkpoint_best_loss=ckpt.get("best_val_loss", None),
                checkpoint_best_dist=ckpt.get("best_val_dist", None),
                crop_size=args.ui_crop_size,
                rollout_steps=args.rollout_steps,
                debug_bbox=args.debug_bbox,
            )
            summary["records"].append(record)

    summary_path = os.path.join(args.output_dir, "picked_trials_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nAll done.")
    print("Saved picked indices:", picked_indices_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()