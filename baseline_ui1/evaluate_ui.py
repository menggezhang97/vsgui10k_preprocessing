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
    build_ui_samples,
    UIBaselineDataset,
)
from model_ui import UIBaselineModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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

        logits = model(cue_id, history_ui_ids, history_xydur)
        loss = criterion(logits, target_ui_id)

        bs = cue_id.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        topk = compute_topk(logits, target_ui_id, ks=(1, 5))
        total_top1 += topk[1] * bs
        total_top5 += topk[5] * bs

        if batch_idx % 100 == 0:
            print(
                f"    eval batch {batch_idx}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"top1={topk[1]:.3f} "
                f"top5={topk[5]:.3f}"
            )

    return {
        "n_samples": int(total_n),
        "loss": total_loss / max(1, total_n),
        "top1": total_top1 / max(1, total_n),
        "top5": total_top5 / max(1, total_n),
    }


def maybe_cap_samples(samples, max_samples):
    if max_samples is None:
        return samples
    return samples[:max_samples]


def main():
    output_dir = "baseline_ui1/outputs/ui_v1_aligned"
    config_path = os.path.join(output_dir, "config.json")
    split_path = os.path.join(output_dir, "split.json")
    cue_vocab_path = os.path.join(output_dir, "cue_vocab.json")
    ui_vocab_path = os.path.join(output_dir, "ui_vocab.json")
    ckpt_path = os.path.join(output_dir, "best.pt")
    sample_stats_path = os.path.join(output_dir, "sample_stats.json")

    cfg = load_json(config_path)
    split_info = load_json(split_path)
    cue_vocab = load_json(cue_vocab_path)
    ui_vocab = load_json(ui_vocab_path)
    saved_sample_stats = load_json(sample_stats_path)

    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    print("output_dir =", output_dir)

    print("loading trials...")
    trials = read_jsonl_gz(cfg["trials_path"])
    trial_index = build_trial_index(trials)

    test_trials = [trial_index[tid] for tid in split_info["test_trial_ids"] if tid in trial_index]
    print("test trials =", len(test_trials))

    print("building test samples...")
    test_samples_full, test_stats = build_ui_samples(
        test_trials,
        cfg["history_len"],
        ui_vocab,
        cue_vocab,
        drop_unknown_target=True,
    )
    print("full test samples =", len(test_samples_full), test_stats)

    test_samples = maybe_cap_samples(test_samples_full, cfg.get("max_test_samples"))
    print("used test samples =", len(test_samples))

    test_ds = UIBaselineDataset(test_samples)
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

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    print("running evaluation...")
    test_result = evaluate(model, test_loader, criterion, device)

    result = {
        "model_name": "simple_ui_sequence_transformer",
        "task_type": "next_ui_token_classification",
        "representation": "history_only_fixation_hit_ui_tokens",
        "n_samples": test_result["n_samples"],
        "loss": test_result["loss"],
        "top1": test_result["top1"],
        "top5": test_result["top5"],
        "ui_vocab_size": len(ui_vocab),
        "cue_vocab_size": len(cue_vocab),
        "history_len": cfg["history_len"],
        "sample_stats": {
            "saved_training_stats": saved_sample_stats,
            "current_test_stats_full": test_stats,
            "current_test_samples_full": len(test_samples_full),
            "current_test_samples_used": len(test_samples),
        },
        "notes": {
            "ui_token_definition": "img_name::ui_class::ui_idx",
            "target_space": "UI token classification space, not fixation (x,y) regression space",
            "mapping_quality_note": "Current UI tokenization depends on coarse auto-segmentation / auto-mapping; UI element boundaries are known to be rough and not finely aligned.",
            "model_scope_note": "This is a history-only UI-sequence baseline, not the final structured UI transformer with full UI memory and cross-attention."
        }
    }

    out_path = os.path.join(output_dir, "eval_test_result.json")
    save_json(result, out_path)

    print("\nEVAL RESULT")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()