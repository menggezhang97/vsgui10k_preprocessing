import os
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

from PIL import Image, ImageDraw, ImageOps

from dataset_crop_decoder import (
    read_jsonl_gz,
    split_trials,
    build_trial_index,
    build_cue_vocab,
    build_seg_index,
    build_ui_type_vocab,
    build_hybrid_coord_samples,
    load_ui_elements_from_seg,
    encode_ui_elements,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pil_to_overlay_with_boxes(
    pil_img: Image.Image,
    ui_geom,
    ui_type_id,
    ui_mask,
    id_to_type: Dict[int, str],
) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    valid_k = int(ui_mask.sum().item())
    for k in range(valid_k):
        xc, yc, w, h = [float(v) for v in ui_geom[k]]
        t_id = int(ui_type_id[k].item())
        t_name = id_to_type.get(t_id, f"<UNK:{t_id}>")

        x1 = (xc - w / 2.0) * W
        y1 = (yc - h / 2.0) * H
        x2 = (xc + w / 2.0) * W
        y2 = (yc + h / 2.0) * H

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), f"{k}:{t_name}", fill="yellow")

    return img


def resize_with_padding(
    img: Image.Image,
    target_size: int = 32,
    fill: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    Keep aspect ratio, resize long side to target_size, then pad to square.
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_size, target_size), fill)

    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def make_grid(
    images: List[Image.Image],
    labels: List[str],
    tile_size: int = 96,
    cols: int = 4,
    bg=(30, 30, 30),
) -> Image.Image:
    if len(images) == 0:
        return Image.new("RGB", (tile_size, tile_size), bg)

    rows = math.ceil(len(images) / cols)
    pad = 8
    label_h = 18

    canvas_w = cols * tile_size + (cols + 1) * pad
    canvas_h = rows * (tile_size + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(images):
        img_rs = img.resize((tile_size, tile_size), Image.NEAREST)
        r = i // cols
        c = i % cols
        x = pad + c * (tile_size + pad)
        y = pad + r * (tile_size + label_h + pad)

        canvas.paste(img_rs, (x, y))
        draw.rectangle([x, y, x + tile_size, y + tile_size], outline="white", width=1)
        draw.text((x + 2, y + tile_size + 2), labels[i], fill="yellow")

    return canvas


def crop_regions_from_kept_elements(
    pil_image: Image.Image,
    kept_elements: List[Dict[str, Any]],
    max_ui_tokens: int,
):
    raw_crops = []
    labels = []

    img_w, img_h = pil_image.size

    for k, elem in enumerate(kept_elements[:max_ui_tokens]):
        x1, y1, x2, y2 = elem["bbox"]
        t_name = elem["type"]

        x1 = int(max(0, min(img_w - 1, round(x1))))
        y1 = int(max(0, min(img_h - 1, round(y1))))
        x2 = int(max(x1 + 1, min(img_w, round(x2))))
        y2 = int(max(y1 + 1, min(img_h, round(y2))))

        crop = pil_image.crop((x1, y1, x2, y2))
        raw_crops.append(crop)
        labels.append(f"{k}:{t_name}")

    return raw_crops, labels


def main():
    parser = argparse.ArgumentParser(description="Inspect raw and padded UI crops")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="ui_crop_inspect_v2")
    parser.add_argument("--crop_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(args.save_dir)

    print("Loading trials...")
    trials = read_jsonl_gz(cfg["trials_path"])
    print("loaded trials =", len(trials))

    train_trials, val_trials, test_trials = split_trials(
        trials,
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg["test_ratio"],
        cfg["seed"],
    )

    if args.split == "train":
        selected_trials = train_trials
    elif args.split == "val":
        selected_trials = val_trials
    else:
        selected_trials = test_trials

    trial_index = build_trial_index(selected_trials)
    seg_index = build_seg_index(cfg["seg_root"])
    cue_vocab = build_cue_vocab(train_trials)
    ui_type_vocab = build_ui_type_vocab(train_trials, seg_index)
    id_to_type = {v: k for k, v in ui_type_vocab.items()}

    samples = build_hybrid_coord_samples(selected_trials, cfg["history_len"])

    # keep one sample per image
    unique_samples = []
    seen = set()
    for s in samples:
        img_name = s["img_name"]
        if img_name not in seen:
            unique_samples.append(s)
            seen.add(img_name)

    samples = unique_samples[:args.num_samples]
    print("n_unique_samples_to_inspect =", len(samples))

    for i, s in enumerate(samples):
        trial = trial_index[s["trial_id"]]
        img_name = s["img_name"]
        img_path = os.path.join(cfg["image_dir"], img_name)

        pil_image = Image.open(img_path).convert("RGB")
        seg_w = float(trial["geom"]["seg_w"])
        seg_h = float(trial["geom"]["seg_h"])

        ui_elements = load_ui_elements_from_seg(seg_index, img_name)

        ui_geom, ui_type_id, ui_mask, kept_elements = encode_ui_elements(
            elements=ui_elements,
            seg_w=seg_w,
            seg_h=seg_h,
            ui_type_vocab=ui_type_vocab,
            max_ui_tokens=cfg["max_ui_tokens"],
            drop_full_screen_root=cfg.get("drop_full_screen_root", True),
        )

        valid_k = int(ui_mask.sum().item())
        print("\n" + "=" * 100)
        print(f"sample {i}")
        print("img_name =", img_name)
        print("valid_k =", valid_k)
        print("ui_types[:10] =", [id_to_type.get(int(x), f'<UNK:{int(x)}>') for x in ui_type_id[:10]])

        # overlay image
        overlay_img = pil_to_overlay_with_boxes(
            pil_img=pil_image,
            ui_geom=ui_geom,
            ui_type_id=ui_type_id,
            ui_mask=ui_mask,
            id_to_type=id_to_type,
        )
        overlay_path = os.path.join(args.save_dir, f"{args.split}_sample_{i}_overlay.png")
        overlay_img.save(overlay_path)
        print("saved overlay ->", overlay_path)

        # raw crops
        raw_crops, labels = crop_regions_from_kept_elements(
            pil_image=pil_image,
            kept_elements=kept_elements,
            max_ui_tokens=cfg["max_ui_tokens"],
        )

        raw_grid = make_grid(
            images=raw_crops,
            labels=labels,
            tile_size=96,
            cols=4,
        )
        raw_grid_path = os.path.join(args.save_dir, f"{args.split}_sample_{i}_raw_crops.png")
        raw_grid.save(raw_grid_path)
        print("saved raw crops ->", raw_grid_path)

        # padded crops
        padded_crops = [resize_with_padding(c, target_size=args.crop_size) for c in raw_crops]
        padded_grid = make_grid(
            images=padded_crops,
            labels=labels,
            tile_size=96,
            cols=4,
        )
        padded_grid_path = os.path.join(args.save_dir, f"{args.split}_sample_{i}_padded_crops.png")
        padded_grid.save(padded_grid_path)
        print("saved padded crops ->", padded_grid_path)

        # optional: save first few individual crops
        sample_dir = os.path.join(args.save_dir, f"{args.split}_sample_{i}_individual")
        ensure_dir(sample_dir)
        for k, (rc, pc, lab) in enumerate(zip(raw_crops[:8], padded_crops[:8], labels[:8])):
            rc.save(os.path.join(sample_dir, f"{k}_raw_{lab.replace(':', '_')}.png"))
            pc.save(os.path.join(sample_dir, f"{k}_padded_{lab.replace(':', '_')}.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()