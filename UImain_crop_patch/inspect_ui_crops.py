import os
import json
import math
import argparse
from typing import Dict, Any

import torch
from PIL import Image, ImageDraw

from dataset_crop_decoder import (
    read_jsonl_gz,
    split_trials,
    build_trial_index,
    build_cue_vocab,
    build_seg_index,
    build_ui_type_vocab,
    build_hybrid_coord_samples,
    HybridPlainDecoderDataset,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    image_tensor: [3, H, W], assumed in [0,1]
    """
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(img)


def draw_ui_boxes(
    pil_img: Image.Image,
    ui_geom: torch.Tensor,
    ui_type_id: torch.Tensor,
    ui_mask: torch.Tensor,
    id_to_type: Dict[int, str],
) -> Image.Image:
    """
    ui_geom: [K, 4] = [xc, yc, w, h] in normalized coordinates
    ui_type_id: [K]
    ui_mask: [K]
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    valid_k = int(ui_mask.sum().item())

    for k in range(valid_k):
        xc, yc, w, h = [float(v) for v in ui_geom[k]]
        type_id = int(ui_type_id[k].item())
        type_name = id_to_type.get(type_id, f"<UNK:{type_id}>")

        x1 = (xc - w / 2.0) * W
        y1 = (yc - h / 2.0) * H
        x2 = (xc + w / 2.0) * W
        y2 = (yc + h / 2.0) * H

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), f"{k}:{type_name}", fill="yellow")

    return img


def make_crop_grid(
    ui_crop_images: torch.Tensor,
    ui_type_id: torch.Tensor,
    ui_mask: torch.Tensor,
    id_to_type: Dict[int, str],
    tile_size: int = 96,
    cols: int = 4,
) -> Image.Image:
    """
    ui_crop_images: [K, 3, Hc, Wc]
    ui_type_id: [K]
    ui_mask: [K]
    """
    valid_k = int(ui_mask.sum().item())
    if valid_k == 0:
        return Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))

    rows = math.ceil(valid_k / cols)
    pad = 8
    label_h = 18

    canvas_w = cols * tile_size + (cols + 1) * pad
    canvas_h = rows * (tile_size + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for k in range(valid_k):
        crop = ui_crop_images[k]  # [3, Hc, Wc]
        type_id = int(ui_type_id[k].item())
        type_name = id_to_type.get(type_id, f"<UNK:{type_id}>")

        crop_pil = tensor_image_to_pil(crop)
        crop_pil = crop_pil.resize((tile_size, tile_size))

        r = k // cols
        c = k % cols

        x = pad + c * (tile_size + pad)
        y = pad + r * (tile_size + label_h + pad)

        canvas.paste(crop_pil, (x, y))
        draw.rectangle([x, y, x + tile_size, y + tile_size], outline="white", width=1)
        draw.text((x + 2, y + tile_size + 2), f"{k}:{type_name}", fill="yellow")

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Inspect cropped UI regions")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to inspect",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="How many unique-image samples to inspect",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ui_crop_inspect",
        help="Directory to save output images",
    )
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

    print(f"Using split = {args.split}, n_trials = {len(selected_trials)}")

    trial_index = build_trial_index(selected_trials)

    print("Building seg index...")
    seg_index = build_seg_index(cfg["seg_root"])
    print("n_seg_files =", len(seg_index))

    print("Building cue vocab...")
    cue_vocab = build_cue_vocab(train_trials)

    print("Building ui type vocab...")
    ui_type_vocab = build_ui_type_vocab(train_trials, seg_index)
    id_to_type = {v: k for k, v in ui_type_vocab.items()}

    print("Building samples...")
    samples = build_hybrid_coord_samples(selected_trials, cfg["history_len"])

    # IMPORTANT: keep only one sample per image so we inspect different GUIs
    unique_samples = []
    seen_imgs = set()
    for s in samples:
        img_name = s["img_name"]
        if img_name not in seen_imgs:
            unique_samples.append(s)
            seen_imgs.add(img_name)

    samples = unique_samples[:args.num_samples]
    print("n_unique_samples_to_inspect =", len(samples))

    dataset = HybridPlainDecoderDataset(
        samples=samples,
        trial_index=trial_index,
        cue_vocab=cue_vocab,
        ui_type_vocab=ui_type_vocab,
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", True),
        crop_size=cfg.get("crop_size", 32),
    )

    for i in range(len(dataset)):
        item = dataset[i]

        image = item["image"]                    # [3, H, W]
        ui_geom = item["ui_geom"]                # [K, 4]
        ui_type_id = item["ui_type_id"]          # [K]
        ui_mask = item["ui_mask"]                # [K]
        ui_crop_images = item["ui_crop_images"]  # [K, 3, crop_size, crop_size]

        valid_k = int(ui_mask.sum().item())

        print("\n" + "=" * 100)
        print(f"sample {i}")
        print("valid_k =", valid_k)
        print("ui_type_id[:10] =", ui_type_id[:10].tolist())
        print("ui_types[:10] =", [id_to_type.get(int(x), f"<UNK:{int(x)}>") for x in ui_type_id[:10]])
        print("ui_geom[:10] =")
        for row in ui_geom[:10]:
            print("   ", [round(float(v), 4) for v in row.tolist()])

        # save overlay image
        pil_img = tensor_image_to_pil(image)
        overlay_img = draw_ui_boxes(
            pil_img=pil_img,
            ui_geom=ui_geom,
            ui_type_id=ui_type_id,
            ui_mask=ui_mask,
            id_to_type=id_to_type,
        )
        overlay_path = os.path.join(args.save_dir, f"{args.split}_sample_{i}_overlay.png")
        overlay_img.save(overlay_path)
        print("saved overlay ->", overlay_path)

        # save crop grid
        crop_grid = make_crop_grid(
            ui_crop_images=ui_crop_images,
            ui_type_id=ui_type_id,
            ui_mask=ui_mask,
            id_to_type=id_to_type,
            tile_size=96,
            cols=4,
        )
        crop_grid_path = os.path.join(args.save_dir, f"{args.split}_sample_{i}_crops.png")
        crop_grid.save(crop_grid_path)
        print("saved crops   ->", crop_grid_path)

    print("\nDone.")


if __name__ == "__main__":
    main()