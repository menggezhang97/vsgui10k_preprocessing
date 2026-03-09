import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 让 train_patch.py 可以 import 同目录文件
sys.path.append(os.path.dirname(__file__))

from datasets import (
    read_jsonl_gz,
    split_trials,
    build_cue_vocab,
    build_patch_samples,
    PatchBaselineDataset,
    save_json,
)
from model_patch import PatchBaselineModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def euclidean_mean(pred, target):
    dist = torch.sqrt(torch.sum((pred - target) ** 2, dim=1))
    return dist.mean().item()


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dist = 0.0
    total_n = 0

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        history = batch["history"].to(device)
        cue_id = batch["cue_id"].to(device)
        target_xy = batch["target_xy"].to(device)

        with torch.set_grad_enabled(train):
            pred_xy = model(image, history, cue_id)
            loss = criterion(pred_xy, target_xy)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_dist += euclidean_mean(pred_xy.detach(), target_xy.detach()) * bs
        total_n += bs
        # print log every 100 batches
        if batch_idx % 100 == 0:
            print(f"    batch {batch_idx}/{len(loader)} loss={loss.item():.6f}")

    return total_loss / total_n, total_dist / total_n


def main():
    config_path = "baseline/configs/patch_baseline_v1.json"
    cfg = load_config(config_path)

    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # 1. read trials
    trials = read_jsonl_gz(cfg["trials_path"])
    print("loaded trials =", len(trials))

    # 2. split
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

    # 3. cue vocab from train only
    cue_vocab = build_cue_vocab(train_trials)
    save_json(cue_vocab, os.path.join(cfg["output_dir"], "cue_vocab.json"))

    # 4. build samples
    train_samples = build_patch_samples(train_trials, cfg["history_len"])
    val_samples = build_patch_samples(val_trials, cfg["history_len"])
    test_samples = build_patch_samples(test_trials, cfg["history_len"])

    print("train samples =", len(train_samples))
    print("val samples   =", len(val_samples))
    print("test samples  =", len(test_samples))

    # 5. datasets
    train_ds = PatchBaselineDataset(
        train_samples,
        cfg["image_dir"],
        cue_vocab,
        cfg["image_size"],
    )
    val_ds = PatchBaselineDataset(
        val_samples,
        cfg["image_dir"],
        cue_vocab,
        cfg["image_size"],
    )
    test_ds = PatchBaselineDataset(
        test_samples,
        cfg["image_dir"],
        cue_vocab,
        cfg["image_size"],
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

    # 6. model
    model = PatchBaselineModel(
        backbone_name=cfg["backbone_name"],
        pretrained=cfg["pretrained"],
        history_len=cfg["history_len"],
        cue_vocab_size=len(cue_vocab),
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # save config copy
    with open(os.path.join(cfg["output_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(cfg["output_dir"], "train_log.csv")
    best_val_loss = math.inf
    best_path = os.path.join(cfg["output_dir"], "best.pt")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dist", "val_loss", "val_dist"])

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, train_dist = run_epoch(
                model, train_loader, optimizer, criterion, device, train=True
            )
            val_loss, val_dist = run_epoch(
                model, val_loader, optimizer, criterion, device, train=False
            )

            writer.writerow([epoch, train_loss, train_dist, val_loss, val_dist])
            f.flush()

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.6f} train_dist={train_dist:.6f} | "
                f"val_loss={val_loss:.6f} val_dist={val_dist:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "cue_vocab": cue_vocab,
                    "best_val_loss": best_val_loss,
                }, best_path)
                print(f"  saved best checkpoint -> {best_path}")

    # 7. final test with best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_dist = run_epoch(
        model, test_loader, optimizer=None, criterion=criterion, device=device, train=False
    )

    result = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mean_euclidean_distance": test_dist,
    }
    save_json(result, os.path.join(cfg["output_dir"], "test_result.json"))

    print("\nFINAL TEST RESULT")
    print(result)


if __name__ == "__main__":
    main()