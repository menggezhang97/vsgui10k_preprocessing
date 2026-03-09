import csv
import json
import math
import os
import random
import sys
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

from dataset_ui import (
    read_jsonl_gz,
    load_json,
    save_json,
    build_trial_index,
    build_cue_vocab,
    build_ui_vocab,
    build_ui_samples,
    UIBaselineDataset,
)
from model_ui import UIBaselineModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def maybe_cap_samples(samples, max_samples):
    if max_samples is None:
        return samples
    return samples[:max_samples]


@torch.no_grad()
def compute_topk(logits, target, ks=(1, 5)):
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    out = {}
    for k in ks:
        c = correct[:k].any(dim=0).float().mean().item()
        out[k] = c
    return out


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for batch_idx, batch in enumerate(loader):
        cue_id = batch["cue_id"].to(device)
        history_ui_ids = batch["history_ui_ids"].to(device)
        history_xydur = batch["history_xydur"].to(device)
        target_ui_id = batch["target_ui_id"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(cue_id, history_ui_ids, history_xydur)
            loss = criterion(logits, target_ui_id)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = cue_id.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        topk = compute_topk(logits.detach(), target_ui_id.detach(), ks=(1, 5))
        total_top1 += topk[1] * bs
        total_top5 += topk[5] * bs

        if batch_idx % 100 == 0:
            print(
                f"    batch {batch_idx}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"top1={topk[1]:.3f} "
                f"top5={topk[5]:.3f}"
            )

    return {
        "loss": total_loss / max(1, total_n),
        "top1": total_top1 / max(1, total_n),
        "top5": total_top5 / max(1, total_n),
    }


def main():
    config_path = "baseline_ui1/configs/ui_baseline_z1.json"
    cfg = load_config(config_path)

    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    print("output_dir =", cfg["output_dir"])

    print("loading trials...")
    trials = read_jsonl_gz(cfg["trials_path"])
    print("loaded trials =", len(trials))

    print("loading split...")
    split_info = load_json(cfg["split_path"])
    trial_index = build_trial_index(trials)

    train_trials = [trial_index[tid] for tid in split_info["train_trial_ids"] if tid in trial_index]
    val_trials = [trial_index[tid] for tid in split_info["val_trial_ids"] if tid in trial_index]
    test_trials = [trial_index[tid] for tid in split_info["test_trial_ids"] if tid in trial_index]

    print("train trials =", len(train_trials))
    print("val trials   =", len(val_trials))
    print("test trials  =", len(test_trials))

    print("building vocab...")
    cue_vocab = build_cue_vocab(train_trials)
    ui_vocab = build_ui_vocab(train_trials)

    save_json(cue_vocab, os.path.join(cfg["output_dir"], "cue_vocab.json"))
    save_json(ui_vocab, os.path.join(cfg["output_dir"], "ui_vocab.json"))
    save_json(split_info, os.path.join(cfg["output_dir"], "split.json"))

    print("building samples...")
    train_samples_full, train_stats = build_ui_samples(
        train_trials,
        cfg["history_len"],
        ui_vocab,
        cue_vocab,
        drop_unknown_target=True,
    )
    val_samples_full, val_stats = build_ui_samples(
        val_trials,
        cfg["history_len"],
        ui_vocab,
        cue_vocab,
        drop_unknown_target=True,
    )
    test_samples_full, test_stats = build_ui_samples(
        test_trials,
        cfg["history_len"],
        ui_vocab,
        cue_vocab,
        drop_unknown_target=True,
    )

    print("full train samples =", len(train_samples_full), train_stats)
    print("full val samples   =", len(val_samples_full), val_stats)
    print("full test samples  =", len(test_samples_full), test_stats)

    train_samples = maybe_cap_samples(train_samples_full, cfg.get("max_train_samples"))
    val_samples = maybe_cap_samples(val_samples_full, cfg.get("max_val_samples"))
    test_samples = maybe_cap_samples(test_samples_full, cfg.get("max_test_samples"))

    print("capped train samples =", len(train_samples))
    print("capped val samples   =", len(val_samples))
    print("capped test samples  =", len(test_samples))

    save_json(
        {
            "train_stats_full": train_stats,
            "val_stats_full": val_stats,
            "test_stats_full": test_stats,
            "train_samples_full": len(train_samples_full),
            "val_samples_full": len(val_samples_full),
            "test_samples_full": len(test_samples_full),
            "train_samples_used": len(train_samples),
            "val_samples_used": len(val_samples),
            "test_samples_used": len(test_samples),
            "max_train_samples": cfg.get("max_train_samples"),
            "max_val_samples": cfg.get("max_val_samples"),
            "max_test_samples": cfg.get("max_test_samples"),
        },
        os.path.join(cfg["output_dir"], "sample_stats.json"),
    )

    train_ds = UIBaselineDataset(train_samples)
    val_ds = UIBaselineDataset(val_samples)
    test_ds = UIBaselineDataset(test_samples)

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
    model = UIBaselineModel(
        ui_vocab_size=len(ui_vocab),
        cue_vocab_size=len(cue_vocab),
        history_len=cfg["history_len"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    with open(os.path.join(cfg["output_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(cfg["output_dir"], "train_log.csv")
    best_val_loss = math.inf
    best_path = os.path.join(cfg["output_dir"], "best.pt")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_top1",
            "train_top5",
            "val_loss",
            "val_top1",
            "val_top5",
        ])

        for epoch in range(1, cfg["epochs"] + 1):
            print(f"\n===== Epoch {epoch:02d}/{cfg['epochs']} =====")

            train_out = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                train=True,
            )
            val_out = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                train=False,
            )

            writer.writerow([
                epoch,
                train_out["loss"],
                train_out["top1"],
                train_out["top5"],
                val_out["loss"],
                val_out["top1"],
                val_out["top5"],
            ])
            f.flush()

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_out['loss']:.6f} "
                f"train_top1={train_out['top1']:.4f} "
                f"train_top5={train_out['top5']:.4f} | "
                f"val_loss={val_out['loss']:.6f} "
                f"val_top1={val_out['top1']:.4f} "
                f"val_top5={val_out['top5']:.4f}"
            )

            if val_out["loss"] < best_val_loss:
                best_val_loss = val_out["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "cue_vocab": cue_vocab,
                        "ui_vocab": ui_vocab,
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )
                print(f"  saved best checkpoint -> {best_path}")

    print("\nloading best checkpoint for final test...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_out = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
    )

    save_json(
        {
            "best_val_loss": best_val_loss,
            "test_loss": test_out["loss"],
            "test_top1": test_out["top1"],
            "test_top5": test_out["top5"],
            "train_samples_used": len(train_samples),
            "val_samples_used": len(val_samples),
            "test_samples_used": len(test_samples),
        },
        os.path.join(cfg["output_dir"], "test_result.json"),
    )

    print("\nFINAL TEST RESULT")
    print(
        {
            "best_val_loss": best_val_loss,
            "test_loss": test_out["loss"],
            "test_top1": test_out["top1"],
            "test_top5": test_out["top5"],
            "train_samples_used": len(train_samples),
            "val_samples_used": len(val_samples),
            "test_samples_used": len(test_samples),
        }
    )


if __name__ == "__main__":
    main()