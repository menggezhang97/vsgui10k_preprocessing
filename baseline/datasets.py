import gzip
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def read_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    trials = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def split_trials(
    trials: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    rng = random.Random(seed)
    trials_copy = trials[:]
    rng.shuffle(trials_copy)

    n = len(trials_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_trials = trials_copy[:n_train]
    val_trials = trials_copy[n_train:n_train + n_val]
    test_trials = trials_copy[n_train + n_val:]
    return train_trials, val_trials, test_trials


def build_cue_vocab(trials: List[Dict[str, Any]]) -> Dict[str, int]:
    cues = sorted({str(t["key"]["cue"]) for t in trials})
    vocab = {"<UNK>": 0}
    for i, cue in enumerate(cues, start=1):
        vocab[cue] = i
    return vocab


def get_duration(p: Dict[str, Any]) -> float:
    for k in ["dur", "duration", "dur_s", "dur_sec", "dt"]:
        if k in p and p[k] is not None:
            return float(p[k])
    return 0.0


def get_xy_seg_norm(p: Dict[str, Any], seg_w: float, seg_h: float) -> Tuple[float, float]:
    x = float(p["xy_seg"]["x"]) / float(seg_w)
    y = float(p["xy_seg"]["y"]) / float(seg_h)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return x, y


def build_patch_samples(
    trials: List[Dict[str, Any]],
    history_len: int,
) -> List[Dict[str, Any]]:
    samples = []

    for t in trials:
        img_name = t["key"]["img_name"]
        cue = str(t["key"]["cue"])
        seg_w = float(t["geom"]["seg_w"])
        seg_h = float(t["geom"]["seg_h"])

        scanpath = t["phases"]["search"]["scanpath"]
        if len(scanpath) < 2:
            continue

        for i in range(1, len(scanpath)):
            hist_points = scanpath[max(0, i - history_len):i]
            target_point = scanpath[i]

            hist_feats = []
            for p in hist_points:
                x, y = get_xy_seg_norm(p, seg_w, seg_h)
                dur = get_duration(p)
                hist_feats.append([x, y, dur])

            while len(hist_feats) < history_len:
                hist_feats.insert(0, [0.0, 0.0, 0.0])

            tx, ty = get_xy_seg_norm(target_point, seg_w, seg_h)

            samples.append({
                "trial_id": t["trial_id"],
                "img_name": img_name,
                "cue": cue,
                "history": hist_feats,
                "target_xy": [tx, ty],
            })

    return samples


class PatchBaselineDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_dir: str,
        cue_vocab: Dict[str, int],
        image_size: int = 224,
    ):
        self.samples = samples
        self.image_dir = image_dir
        self.cue_vocab = cue_vocab

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        img_path = os.path.join(self.image_dir, s["img_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)

        cue_id = self.cue_vocab.get(s["cue"], 0)

        history = torch.tensor(s["history"], dtype=torch.float32)
        target_xy = torch.tensor(s["target_xy"], dtype=torch.float32)

        return {
            "image": image,
            "history": history,
            "cue_id": torch.tensor(cue_id, dtype=torch.long),
            "target_xy": target_xy,
        }