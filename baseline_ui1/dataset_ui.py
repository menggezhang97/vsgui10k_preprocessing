import gzip
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def read_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    trials = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_trial_index(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["trial_id"]: t for t in trials}


def build_cue_vocab(trials: List[Dict[str, Any]]) -> Dict[str, int]:
    cues = sorted({str(t["key"]["cue"]) for t in trials})
    vocab = {"<UNK>": 0}
    for i, cue in enumerate(cues, start=1):
        vocab[cue] = i
    return vocab


def ui_token_from_fixation(trial: Dict[str, Any], p: Dict[str, Any]) -> str:
    """
    Use mapped UI info already stored in trial export.
    Return special token <NONE> if not mapped.
    """
    ui = p.get("ui", None) or {}
    ui_class = ui.get("class", None)
    ui_idx = ui.get("idx", None)

    if ui_class is None or ui_idx is None:
        return "<NONE>"

    img_name = str(trial["key"]["img_name"])
    return f"{img_name}::{ui_class}::{int(ui_idx)}"


def build_ui_vocab(train_trials: List[Dict[str, Any]]) -> Dict[str, int]:
    vocab = {
        "<PAD>": 0,
        "<NONE>": 1,
        "<UNK>": 2,
    }

    next_id = 3
    for t in train_trials:
        scanpath = t["phases"]["search"]["scanpath"]
        for p in scanpath:
            tok = ui_token_from_fixation(t, p)
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
    return vocab


def get_xy_seg_norm(p: Dict[str, Any], seg_w: float, seg_h: float) -> Tuple[float, float]:
    x = float(p["xy_seg"]["x"]) / float(seg_w)
    y = float(p["xy_seg"]["y"]) / float(seg_h)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return x, y


def get_duration(p: Dict[str, Any]) -> float:
    if "dur" in p and p["dur"] is not None:
        return float(p["dur"])
    return 0.0


def build_ui_samples(
    trials: List[Dict[str, Any]],
    history_len: int,
    ui_vocab: Dict[str, int],
    cue_vocab: Dict[str, int],
    drop_unknown_target: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build samples for next-UI-token prediction.

    Input:
      - history of previous UI tokens
      - optional history xy/dur
      - cue id

    Label:
      - next UI token id

    We skip target = <NONE>, because predicting "no mapped UI" is not very useful
    for this first structured baseline.
    """
    samples = []
    stats = {
        "total_candidates": 0,
        "kept": 0,
        "skipped_target_none": 0,
        "skipped_target_unknown": 0,
    }

    for t in trials:
        cue = str(t["key"]["cue"])
        cue_id = cue_vocab.get(cue, 0)

        seg_w = float(t["geom"]["seg_w"])
        seg_h = float(t["geom"]["seg_h"])

        scanpath = t["phases"]["search"]["scanpath"]
        if len(scanpath) < 2:
            continue

        token_seq = [ui_token_from_fixation(t, p) for p in scanpath]

        for i in range(1, len(scanpath)):
            stats["total_candidates"] += 1

            target_tok = token_seq[i]
            if target_tok == "<NONE>":
                stats["skipped_target_none"] += 1
                continue

            target_id = ui_vocab.get(target_tok, ui_vocab["<UNK>"])
            if drop_unknown_target and target_id == ui_vocab["<UNK>"]:
                stats["skipped_target_unknown"] += 1
                continue

            hist_points = scanpath[max(0, i - history_len):i]
            hist_tokens = token_seq[max(0, i - history_len):i]

            hist_token_ids = [ui_vocab.get(tok, ui_vocab["<UNK>"]) for tok in hist_tokens]
            hist_xydur = []

            for p in hist_points:
                x, y = get_xy_seg_norm(p, seg_w, seg_h)
                dur = get_duration(p)
                hist_xydur.append([x, y, dur])

            while len(hist_token_ids) < history_len:
                hist_token_ids.insert(0, ui_vocab["<PAD>"])
                hist_xydur.insert(0, [0.0, 0.0, 0.0])

            samples.append({
                "trial_id": t["trial_id"],
                "cue_id": cue_id,
                "history_ui_ids": hist_token_ids,
                "history_xydur": hist_xydur,
                "target_ui_id": target_id,
            })
            stats["kept"] += 1

    return samples, stats


class UIBaselineDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "cue_id": torch.tensor(s["cue_id"], dtype=torch.long),
            "history_ui_ids": torch.tensor(s["history_ui_ids"], dtype=torch.long),
            "history_xydur": torch.tensor(s["history_xydur"], dtype=torch.float32),
            "target_ui_id": torch.tensor(s["target_ui_id"], dtype=torch.long),
        }