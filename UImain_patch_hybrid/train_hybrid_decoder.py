import csv
import json
import math
import os
from pyexpat import model
import random
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

from dataset_hybrid_decoder import (
    read_jsonl_gz,
    split_trials,
    build_trial_index,
    build_cue_vocab,
    build_seg_index,
    build_ui_type_vocab,
    build_hybrid_coord_samples,
    HybridPlainDecoderDataset,
    save_json,
)
from model_UImain_hybrid import HybridPlainDecoderModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config json file"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scale factor for UI memory in hybrid fusion (default: 1.0)",
    )

    parser.add_argument(
        "--freeze_patch",
        action="store_true",
        help="Freeze patch backbone if set",
    )

    return parser.parse_args()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def euclidean_mean(pred, target):
    dist = torch.sqrt(torch.sum((pred - target) ** 2, dim=1))
    return dist.mean().item()


def run_epoch(model, loader, optimizer, criterion, device, train=True, debug_print=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dist = 0.0
    total_n = 0

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        cue_id = batch["cue_id"].to(device)
        history_xydur = batch["history_xydur"].to(device)
        target_xy = batch["target_xy"].to(device)

        ui_geom = batch["ui_geom"].to(device)
        ui_type_id = batch["ui_type_id"].to(device)
        ui_mask = batch["ui_mask"].to(device)

        if debug_print and batch_idx == 0:
            print("image.shape =", image.shape)
            print("cue_id.shape =", cue_id.shape)
            print("history_xydur.shape =", history_xydur.shape)
            print("target_xy.shape =", target_xy.shape)
            print("ui_geom.shape =", ui_geom.shape)
            print("ui_type_id.shape =", ui_type_id.shape)
            print("ui_mask.shape =", ui_mask.shape)
            print("ui valid tokens per sample =", ui_mask.sum(dim=1))
            print("ui_geom[0, :5] =", ui_geom[0, :5])
            print("ui_type_id[0, :10] =", ui_type_id[0, :10])
            print("target_xy[:5] =", target_xy[:5])

        with torch.set_grad_enabled(train):
            pred_xy = model(
                image=image,
                cue_id=cue_id,
                history_xydur=history_xydur,
                ui_geom=ui_geom,
                ui_type_id=ui_type_id,
                ui_mask=ui_mask,
            )
            loss = criterion(pred_xy, target_xy)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = cue_id.size(0)
        total_loss += loss.item() * bs
        total_dist += euclidean_mean(pred_xy.detach(), target_xy.detach()) * bs
        total_n += bs

        if batch_idx % 20 == 0:
            print(f"    batch {batch_idx}/{len(loader)} loss={loss.item():.6f}")

    return total_loss / total_n, total_dist / total_n


def save_checkpoint(
    path,
    model,
    cfg,
    cue_vocab,
    ui_type_vocab,
    epoch,
    current_val_loss,
    current_val_dist,
    best_val_loss,
    best_val_dist,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "cue_vocab": cue_vocab,
            "ui_type_vocab": ui_type_vocab,
            "epoch": epoch,
            "current_val_loss": current_val_loss,
            "current_val_dist": current_val_dist,
            "best_val_loss": best_val_loss,
            "best_val_dist": best_val_dist,
        },
        path,
    )


def main():
    args = parse_args()
    config_path = args.config
    print("config_path =", config_path)
    cfg = load_config(config_path)

    # override from command line
    cfg["ui_memory_scale"] = args.alpha
    cfg["freeze_patch_backbone"] = args.freeze_patch

    print("ui_memory_scale =", cfg["ui_memory_scale"])
    print("freeze_patch_backbone =", cfg["freeze_patch_backbone"])    

    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    print("output_dir =", cfg["output_dir"])

    trials = read_jsonl_gz(cfg["trials_path"])
    print("loaded trials =", len(trials))

    print("splitting trials...")
    train_trials, val_trials, test_trials = split_trials(
        trials,
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg["test_ratio"],
        cfg["seed"],
    )

    split_info = {
        "train_trial_ids": [t["trial_id"] for t in train_trials],
        "val_trial_ids": [t["trial_id"] for t in val_trials],
        "test_trial_ids": [t["trial_id"] for t in test_trials],
    }
    save_json(split_info, os.path.join(cfg["output_dir"], "split.json"))

    print("building trial index...")
    train_trial_index = build_trial_index(train_trials)
    val_trial_index = build_trial_index(val_trials)
    test_trial_index = build_trial_index(test_trials)

    print("building segmentation index...")
    seg_index = build_seg_index(cfg["seg_root"])
    print("n_seg_files =", len(seg_index))

    print("building cue vocab...")
    cue_vocab = build_cue_vocab(train_trials)

    print("building ui type vocab...")
    ui_type_vocab = build_ui_type_vocab(train_trials, seg_index)

    save_json(cue_vocab, os.path.join(cfg["output_dir"], "cue_vocab.json"))
    save_json(ui_type_vocab, os.path.join(cfg["output_dir"], "ui_type_vocab.json"))

    print("building samples...")
    train_samples = build_hybrid_coord_samples(train_trials, cfg["history_len"])
    val_samples = build_hybrid_coord_samples(val_trials, cfg["history_len"])
    test_samples = build_hybrid_coord_samples(test_trials, cfg["history_len"])

    if cfg.get("max_train_samples") is not None:
        train_samples = train_samples[:cfg["max_train_samples"]]
    if cfg.get("max_val_samples") is not None:
        val_samples = val_samples[:cfg["max_val_samples"]]
    if cfg.get("max_test_samples") is not None:
        test_samples = test_samples[:cfg["max_test_samples"]]

    print("train samples =", len(train_samples))
    print("val samples   =", len(val_samples))
    print("test samples  =", len(test_samples))

    print("building datasets...")
    train_ds = HybridPlainDecoderDataset(
        samples=train_samples,
        trial_index=train_trial_index,
        cue_vocab=cue_vocab,
        ui_type_vocab=ui_type_vocab,
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
    )
    val_ds = HybridPlainDecoderDataset(
        samples=val_samples,
        trial_index=val_trial_index,
        cue_vocab=cue_vocab,
        ui_type_vocab=ui_type_vocab,
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
    )
    test_ds = HybridPlainDecoderDataset(
        samples=test_samples,
        trial_index=test_trial_index,
        cue_vocab=cue_vocab,
        ui_type_vocab=ui_type_vocab,
        seg_index=seg_index,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
        max_ui_tokens=cfg["max_ui_tokens"],
        drop_full_screen_root=cfg.get("drop_full_screen_root", False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    print("loading model...")
    model = HybridPlainDecoderModel(
        vit_name=cfg["vit_name"],
        pretrained=cfg["pretrained"],
        cue_vocab_size=len(cue_vocab),
        ui_type_vocab_size=len(ui_type_vocab),
        history_len=cfg["history_len"],
        ui_geom_dim=cfg["ui_geom_dim"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        ui_memory_scale=cfg["ui_memory_scale"],
        freeze_patch_backbone=cfg["freeze_patch_backbone"],
    ).to(device)
    print("model.ui_memory_scale =", model.ui_memory_scale)
    print("model.freeze_patch_backbone =", model.freeze_patch_backbone)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("trainable_params =", trainable_params)
    print("total_params     =", total_params)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg["lr"],
    weight_decay=cfg["weight_decay"],
    )

    with open(os.path.join(cfg["output_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(cfg["output_dir"], "train_log.csv")
    history_json_path = os.path.join(cfg["output_dir"], "metrics_history.json")

    best_val_loss = math.inf
    best_val_dist = math.inf

    best_path = os.path.join(cfg["output_dir"], "best.pt")
    best_loss_path = os.path.join(cfg["output_dir"], "best_loss.pt")
    best_dist_path = os.path.join(cfg["output_dir"], "best_dist.pt")
    last_path = os.path.join(cfg["output_dir"], "last.pt")

    metrics_history = []

    print("starting training...")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dist", "val_loss", "val_dist"])

        for epoch in range(1, cfg["epochs"] + 1):
            print(f"\n===== Epoch {epoch:02d}/{cfg['epochs']} =====")

            train_loss, train_dist = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                train=True,
                debug_print=cfg.get("debug_print", False),
            )
            val_loss, val_dist = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                train=False,
                debug_print=cfg.get("debug_print", False),
            )

            writer.writerow([epoch, train_loss, train_dist, val_loss, val_dist])
            f.flush()

            metrics_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dist": train_dist,
                    "val_loss": val_loss,
                    "val_dist": val_dist,
                }
            )
            save_json(metrics_history, history_json_path)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.6f} train_dist={train_dist:.6f} | "
                f"val_loss={val_loss:.6f} val_dist={val_dist:.6f}"
            )

            epoch_ckpt_path = os.path.join(cfg["output_dir"], f"epoch_{epoch:02d}.pt")
            save_checkpoint(
                epoch_ckpt_path,
                model,
                cfg,
                cue_vocab,
                ui_type_vocab,
                epoch,
                val_loss,
                val_dist,
                best_val_loss,
                best_val_dist,
            )
            print(f"  saved epoch checkpoint -> {epoch_ckpt_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path,
                    model,
                    cfg,
                    cue_vocab,
                    ui_type_vocab,
                    epoch,
                    val_loss,
                    val_dist,
                    best_val_loss,
                    best_val_dist,
                )
                save_checkpoint(
                    best_loss_path,
                    model,
                    cfg,
                    cue_vocab,
                    ui_type_vocab,
                    epoch,
                    val_loss,
                    val_dist,
                    best_val_loss,
                    best_val_dist,
                )
                print(f"  saved best checkpoint (by val_loss) -> {best_path}")
                print(f"  saved best_loss checkpoint         -> {best_loss_path}")

            if val_dist < best_val_dist:
                best_val_dist = val_dist
                save_checkpoint(
                    best_dist_path,
                    model,
                    cfg,
                    cue_vocab,
                    ui_type_vocab,
                    epoch,
                    val_loss,
                    val_dist,
                    best_val_loss,
                    best_val_dist,
                )
                print(f"  saved best_dist checkpoint         -> {best_dist_path}")

    save_checkpoint(
        last_path,
        model,
        cfg,
        cue_vocab,
        ui_type_vocab,
        cfg["epochs"],
        val_loss,
        val_dist,
        best_val_loss,
        best_val_dist,
    )
    print(f"\nsaved last checkpoint -> {last_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_dist = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        debug_print=cfg.get("debug_print", False),
    )

    result = {
        "selection_rule": "best_val_loss",
        "selected_epoch": ckpt.get("epoch", None),
        "best_val_loss": best_val_loss,
        "best_val_dist": best_val_dist,
        "test_loss": test_loss,
        "test_dist": test_dist,
        "train_samples_used": len(train_samples),
        "val_samples_used": len(val_samples),
        "test_samples_used": len(test_samples),
    }
    save_json(result, os.path.join(cfg["output_dir"], "test_result.json"))

    print("\nFINAL TEST RESULT")
    print(result)


if __name__ == "__main__":
    main()