import json
import math
import os
import sys
import random
import gzip
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# allow local imports
sys.path.append(os.path.dirname(__file__))
from model_patch import PatchBaselineModel


# =========================
# basic utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    trials = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def point_in_bbox(x: float, y: float, x0: float, y0: float, x1: float, y1: float) -> bool:
    return (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)


def expand_bbox(x0, y0, x1, y1, r, W, H):
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = (x1 - x0) * r, (y1 - y0) * r
    nx0 = clamp(cx - 0.5 * w, 0, W)
    nx1 = clamp(cx + 0.5 * w, 0, W)
    ny0 = clamp(cy - 0.5 * h, 0, H)
    ny1 = clamp(cy + 0.5 * h, 0, H)
    return nx0, ny0, nx1, ny1


# =========================
# schema helpers
# =========================
def build_trial_index(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["trial_id"]: t for t in trials}


def get_duration(p: Dict[str, Any]) -> float:
    for k in ["dur", "duration", "dur_s", "dur_sec", "dt"]:
        if k in p and p[k] is not None:
            return float(p[k])
    return 0.0


def get_xy_seg_norm(p: Dict[str, Any], seg_w: float, seg_h: float) -> Tuple[float, float]:
    x = float(p["xy_seg"]["x"]) / float(seg_w)
    y = float(p["xy_seg"]["y"]) / float(seg_h)
    x = clamp(x, 0.0, 1.0)
    y = clamp(y, 0.0, 1.0)
    return x, y


def build_cue_vocab_from_json(path: str) -> Dict[str, int]:
    raw = load_json(path)
    # json may store int values as numbers, keep as dict[str,int]
    return {str(k): int(v) for k, v in raw.items()}


# =========================
# evaluation sample builder
# =========================
def build_patch_eval_samples(
    trials: List[Dict[str, Any]],
    history_len: int,
    cue_vocab: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Build next-fixation prediction samples for evaluation.
    Includes metadata needed for bbox hit-rate evaluation.
    """
    samples = []

    for t in trials:
        img_name = t["key"]["img_name"]
        cue = str(t["key"]["cue"])
        cue_id = cue_vocab.get(cue, 0)

        seg_w = float(t["geom"]["seg_w"])
        seg_h = float(t["geom"]["seg_h"])

        absent = int(t["key"]["absent"])

        bbox = t["target"].get("bbox_seg", None)
        bbox_seg = None
        if bbox is not None:
            bbox_seg = [
                float(bbox["x0"]),
                float(bbox["y0"]),
                float(bbox["x1"]),
                float(bbox["y1"]),
            ]

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
                "cue_id": cue_id,
                "history": hist_feats,
                "target_xy": [tx, ty],
                "seg_w": seg_w,
                "seg_h": seg_h,
                "absent": absent,
                "bbox_seg": bbox_seg,
            })

    return samples


class PatchEvalDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], image_dir: str, image_size: int = 224):
        self.samples = samples
        self.image_dir = image_dir
        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        img_path = os.path.join(self.image_dir, s["img_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)

        target_xy = torch.tensor(s["target_xy"], dtype=torch.float32)
        history = torch.tensor(s["history"], dtype=torch.float32)
        cue_id = torch.tensor(s["cue_id"], dtype=torch.long)

        bbox_seg = s["bbox_seg"] if s["bbox_seg"] is not None else [-1.0, -1.0, -1.0, -1.0]

        return {
            "image": image,
            "history": history,
            "cue_id": cue_id,
            "target_xy": target_xy,
            "seg_w": torch.tensor(s["seg_w"], dtype=torch.float32),
            "seg_h": torch.tensor(s["seg_h"], dtype=torch.float32),
            "absent": torch.tensor(s["absent"], dtype=torch.long),
            "bbox_seg": torch.tensor(bbox_seg, dtype=torch.float32),
        }


# =========================
# evaluation
# =========================
@torch.no_grad()
def evaluate(model, loader, device, r_list=(1.0, 1.5, 2.0, 3.0)):
    model.eval()

    norm_dists = []
    px_dists = []

    present_counts = {r: 0 for r in r_list}
    present_hits = {r: 0 for r in r_list}

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        history = batch["history"].to(device)
        cue_id = batch["cue_id"].to(device)
        target_xy = batch["target_xy"].to(device)

        seg_w = batch["seg_w"].to(device)
        seg_h = batch["seg_h"].to(device)
        absent = batch["absent"].to(device)
        bbox_seg = batch["bbox_seg"].to(device)

        pred_xy = model(image, history, cue_id)

        # normalized Euclidean distance
        diff = pred_xy - target_xy
        dist_norm = torch.sqrt(torch.sum(diff * diff, dim=1))  # [B]
        norm_dists.extend(dist_norm.cpu().tolist())

        # segmentation-space pixel distance
        dx_px = (pred_xy[:, 0] - target_xy[:, 0]) * seg_w
        dy_px = (pred_xy[:, 1] - target_xy[:, 1]) * seg_h
        dist_px = torch.sqrt(dx_px * dx_px + dy_px * dy_px)
        px_dists.extend(dist_px.cpu().tolist())

        # target bbox hit rate (present-only)
        pred_x_seg = pred_xy[:, 0] * seg_w
        pred_y_seg = pred_xy[:, 1] * seg_h

        for i in range(pred_xy.size(0)):
            if int(absent[i].item()) == 1:
                continue

            x0, y0, x1, y1 = [float(v) for v in bbox_seg[i].tolist()]
            W = float(seg_w[i].item())
            H = float(seg_h[i].item())

            # skip if bbox missing
            if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
                continue

            px = float(pred_x_seg[i].item())
            py = float(pred_y_seg[i].item())

            for r in r_list:
                ex0, ey0, ex1, ey1 = expand_bbox(x0, y0, x1, y1, r, W, H)
                present_counts[r] += 1
                if point_in_bbox(px, py, ex0, ey0, ex1, ey1):
                    present_hits[r] += 1

        if batch_idx % 100 == 0:
            print(f"    eval batch {batch_idx}/{len(loader)}")

    result = {
        "n_samples": int(len(norm_dists)),
        "normalized_distance": {
            "mean": float(np.mean(norm_dists)) if norm_dists else None,
            "median": float(np.median(norm_dists)) if norm_dists else None,
        },
        "pixel_distance_seg_space": {
            "mean": float(np.mean(px_dists)) if px_dists else None,
            "median": float(np.median(px_dists)) if px_dists else None,
        },
        "target_hit_rate_present_only": {},
    }

    for r in r_list:
        n = present_counts[r]
        hit = present_hits[r]
        result["target_hit_rate_present_only"][str(r)] = {
            "n_present_samples": int(n),
            "hit_rate": float(hit / n) if n > 0 else None,
        }

    return result


def main():
    output_dir = "baseline/outputs/patch_v1"
    config_path = os.path.join(output_dir, "config.json")
    split_path = os.path.join(output_dir, "split.json")
    cue_vocab_path = os.path.join(output_dir, "cue_vocab.json")
    ckpt_path = os.path.join(output_dir, "best.pt")

    cfg = load_json(config_path)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    print("loading trials...")
    trials = read_jsonl_gz(cfg["trials_path"])
    trial_index = build_trial_index(trials)

    split_info = load_json(split_path)
    cue_vocab = build_cue_vocab_from_json(cue_vocab_path)

    test_trials = [trial_index[tid] for tid in split_info["test_trial_ids"] if tid in trial_index]
    print("test trials =", len(test_trials))

    print("building evaluation samples...")
    test_samples = build_patch_eval_samples(
        test_trials,
        history_len=cfg["history_len"],
        cue_vocab=cue_vocab,
    )
    print("test samples =", len(test_samples))

    test_ds = PatchEvalDataset(
        samples=test_samples,
        image_dir=cfg["image_dir"],
        image_size=cfg["image_size"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    print("loading model...")
    model = PatchBaselineModel(
        backbone_name=cfg["backbone_name"],
        pretrained=False,
        history_len=cfg["history_len"],
        cue_vocab_size=len(cue_vocab),
        dropout=cfg["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("running evaluation...")
    result = evaluate(model, test_loader, device=device, r_list=(1.0, 1.5, 2.0, 3.0))

    out_path = os.path.join(output_dir, "eval_test_result.json")
    save_json(result, out_path)

    print("\nEVAL RESULT")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()