import os
import json
import argparse
from typing import Dict, Any

import torch
from PIL import Image, ImageDraw

from dataset_hybrid_decoder import (
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


def inspect_dataset(
    cfg: Dict[str, Any],
    split_name: str,
    num_samples: int,
    save_dir: str,
):
    ensure_dir(save_dir)

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

    if split_name == "train":
        selected_trials = train_trials
    elif split_name == "val":
        selected_trials = val_trials
    else:
        selected_trials = test_trials

    print(f"Using split = {split_name}, n_trials = {len(selected_trials)}")

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

    unique_samples = []
    seen_imgs = set()

    for s in samples:
        img_name = s["img_name"]
        if img_name not in seen_imgs:
            unique_samples.append(s)
            seen_imgs.add(img_name)

    samples = unique_samples[:num_samples]
    print("n_samples_to_inspect =", len(samples))

    dataset = HybridPlainDecoderDataset(
        samples=samples,
        trial_index=trial_index,
        cue_vocab=cue_vocab,
        ui_type_vocab=ui_type_vocab,
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
    )

    for i in range(len(dataset)):
        item = dataset[i]

        image = item["image"]              # [3, H, W]
        cue_id = item["cue_id"]
        history_xydur = item["history_xydur"]
        target_xy = item["target_xy"]
        ui_geom = item["ui_geom"]          # [K, 4]
        ui_type_id = item["ui_type_id"]    # [K]
        ui_mask = item["ui_mask"]          # [K]

        valid_k = int(ui_mask.sum().item())

        print("\n" + "=" * 100)
        print(f"sample {i}")
        print("cue_id =", int(cue_id.item()) if torch.is_tensor(cue_id) else cue_id)
        print("history_xydur.shape =", tuple(history_xydur.shape))
        print("target_xy =", [round(float(v), 4) for v in target_xy.tolist()])
        print("valid_k =", valid_k)
        print("ui_type_id[:10] =", ui_type_id[:10].tolist())
        print("ui_types[:10] =", [id_to_type.get(int(x), f"<UNK:{int(x)}>") for x in ui_type_id[:10]])
        print("ui_geom[:10] =")
        for row in ui_geom[:10]:
            print("   ", [round(float(v), 4) for v in row.tolist()])

        pil_img = tensor_image_to_pil(image)
        vis_img = draw_ui_boxes(
            pil_img=pil_img,
            ui_geom=ui_geom,
            ui_type_id=ui_type_id,
            ui_mask=ui_mask,
            id_to_type=id_to_type,
        )

        save_path = os.path.join(save_dir, f"{split_name}_sample_{i}.png")
        vis_img.save(save_path)
        print("saved image ->", save_path)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Inspect current UI tokens")
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
        help="How many samples to inspect",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ui_token_inspect",
        help="Directory to save visualization images",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    inspect_dataset(
        cfg=cfg,
        split_name=args.split,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()