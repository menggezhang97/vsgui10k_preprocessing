import gzip
import json
import os
import random
from typing import Any, Dict, List, Tuple, Optional

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


def split_trials(
    trials: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
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


def build_trial_index(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["trial_id"]: t for t in trials}


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


def normalize_ui_type(raw_type: Any) -> str:
    if raw_type is None:
        return "<UNK>"
    s = str(raw_type).strip()
    if not s:
        return "<UNK>"
    return s


def strip_ext(filename: str) -> str:
    base, _ = os.path.splitext(filename)
    return base


def build_seg_index(seg_root: str) -> Dict[str, str]:
    """
    Build mapping:
      image stem -> segmentation json full path
    Example:
      "42962f" -> ".../data/segmentation/block_0/42962f.json"
    """
    seg_index = {}
    for root, _, files in os.walk(seg_root):
        for fn in files:
            if fn.lower().endswith(".json"):
                stem = os.path.splitext(fn)[0]
                seg_index[stem] = os.path.join(root, fn)
    return seg_index


def find_seg_path(seg_index: Dict[str, str], img_name: str) -> str:
    stem = strip_ext(img_name)
    if stem in seg_index:
        return seg_index[stem]
    raise FileNotFoundError(f"Cannot find segmentation file for image: {img_name}")


def parse_bbox_from_elem(elem: Dict[str, Any]) -> Optional[List[float]]:
    """
    Return [x1, y1, x2, y2]
    """
    bbox = elem.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        if float(x2) > float(x1) and float(y2) > float(y1):
            return [float(x1), float(y1), float(x2), float(y2)]
        return [float(x1), float(y1), float(x1) + float(x2), float(y1) + float(y2)]

    bbox_xywh = elem.get("bbox_xywh")
    if isinstance(bbox_xywh, list) and len(bbox_xywh) == 4:
        x, y, w, h = bbox_xywh
        return [float(x), float(y), float(x) + float(w), float(y) + float(h)]

    if all(k in elem for k in ["x1", "y1", "x2", "y2"]):
        return [
            float(elem["x1"]), float(elem["y1"]),
            float(elem["x2"]), float(elem["y2"]),
        ]

    if all(k in elem for k in ["x", "y", "w", "h"]):
        x = float(elem["x"])
        y = float(elem["y"])
        w = float(elem["w"])
        h = float(elem["h"])
        return [x, y, x + w, y + h]

    if all(k in elem for k in ["column_min", "row_min", "column_max", "row_max"]):
        return [
            float(elem["column_min"]), float(elem["row_min"]),
            float(elem["column_max"]), float(elem["row_max"]),
        ]

    if "position" in elem and "size" in elem:
        pos = elem["position"]
        size = elem["size"]
        if isinstance(pos, dict) and isinstance(size, dict):
            if all(k in pos for k in ["x", "y"]) and all(k in size for k in ["width", "height"]):
                x = float(pos["x"])
                y = float(pos["y"])
                w = float(size["width"])
                h = float(size["height"])
                return [x, y, x + w, y + h]

    return None


def load_ui_elements_from_seg(seg_index: Dict[str, str], img_name: str) -> List[Dict[str, Any]]:
    seg_path = find_seg_path(seg_index, img_name)
    data = load_json(seg_path)

    elements = (
        data.get("elements")
        or data.get("ui_elements")
        or data.get("compos")
        or data.get("components")
        or data.get("children")
        or []
    )

    out = []
    for elem in elements:
        if not isinstance(elem, dict):
            continue

        bbox = parse_bbox_from_elem(elem)
        if bbox is None:
            continue

        ui_type = (
            elem.get("class")
            or elem.get("type")
            or elem.get("category")
            or elem.get("label")
            or elem.get("component_label")
            or "<UNK>"
        )

        out.append({
            "bbox": bbox,
            "type": normalize_ui_type(ui_type),
        })

    return out


def build_ui_type_vocab(train_trials: List[Dict[str, Any]], seg_index: Dict[str, str]) -> Dict[str, int]:
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    next_id = 2

    for i, t in enumerate(train_trials):
        if i % 500 == 0:
            print(f"  build_ui_type_vocab progress: {i}/{len(train_trials)}")

        img_name = t["key"]["img_name"]
        try:
            elems = load_ui_elements_from_seg(seg_index, img_name)
        except FileNotFoundError:
            elems = []

        for e in elems:
            ui_type = normalize_ui_type(e.get("type", "<UNK>"))
            if ui_type not in vocab:
                vocab[ui_type] = next_id
                next_id += 1

    return vocab


def encode_ui_elements(
    elements: List[Dict[str, Any]],
    seg_w: float,
    seg_h: float,
    ui_type_vocab: Dict[str, int],
    max_ui_tokens: int,
    drop_full_screen_root: bool = False,
):
    """
    geometry feature:
      [xc, yc, w, h]
    normalized to [0,1]
    """
    feats = []
    type_ids = []

    for elem in elements:
        x1, y1, x2, y2 = elem["bbox"]

        x1 = max(0.0, min(float(seg_w), float(x1)))
        y1 = max(0.0, min(float(seg_h), float(y1)))
        x2 = max(0.0, min(float(seg_w), float(x2)))
        y2 = max(0.0, min(float(seg_h), float(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) / 2.0) / float(seg_w)
        yc = ((y1 + y2) / 2.0) / float(seg_h)
        w = (x2 - x1) / float(seg_w)
        h = (y2 - y1) / float(seg_h)

        if drop_full_screen_root and w > 0.95 and h > 0.95:
            continue

        feats.append([xc, yc, w, h])

        ui_type = normalize_ui_type(elem.get("type", "<UNK>"))
        type_ids.append(ui_type_vocab.get(ui_type, ui_type_vocab["<UNK>"]))

        if len(feats) >= max_ui_tokens:
            break

    valid_len = len(feats)

    while len(feats) < max_ui_tokens:
        feats.append([0.0, 0.0, 0.0, 0.0])
        type_ids.append(ui_type_vocab["<PAD>"])

    mask = [1] * valid_len + [0] * (max_ui_tokens - valid_len)

    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(type_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
    )


def build_hybrid_coord_samples(
    trials: List[Dict[str, Any]],
    history_len: int,
):
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
                "history_xydur": hist_feats,
                "target_xy": [tx, ty],
            })

    return samples


class HybridPlainDecoderDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        trial_index: Dict[str, Dict[str, Any]],
        cue_vocab: Dict[str, int],
        ui_type_vocab: Dict[str, int],
        seg_index: Dict[str, str],
        image_dir: str,
        image_size: int = 224,
        max_ui_tokens: int = 64,
        drop_full_screen_root: bool = False,
    ):
        self.samples = samples
        self.trial_index = trial_index
        self.cue_vocab = cue_vocab
        self.ui_type_vocab = ui_type_vocab
        self.seg_index = seg_index
        self.image_dir = image_dir
        self.image_size = image_size
        self.max_ui_tokens = max_ui_tokens
        self.drop_full_screen_root = drop_full_screen_root

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        trial = self.trial_index[s["trial_id"]]

        cue_id = self.cue_vocab.get(s["cue"], 0)
        history_xydur = torch.tensor(s["history_xydur"], dtype=torch.float32)
        target_xy = torch.tensor(s["target_xy"], dtype=torch.float32)

        img_path = os.path.join(self.image_dir, s["img_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)

        seg_w = float(trial["geom"]["seg_w"])
        seg_h = float(trial["geom"]["seg_h"])

        ui_elements = load_ui_elements_from_seg(self.seg_index, s["img_name"])
        ui_geom, ui_type_id, ui_mask = encode_ui_elements(
            elements=ui_elements,
            seg_w=seg_w,
            seg_h=seg_h,
            ui_type_vocab=self.ui_type_vocab,
            max_ui_tokens=self.max_ui_tokens,
            drop_full_screen_root=self.drop_full_screen_root,
        )

        return {
            "image": image,
            "cue_id": torch.tensor(cue_id, dtype=torch.long),
            "history_xydur": history_xydur,
            "target_xy": target_xy,
            "ui_geom": ui_geom,
            "ui_type_id": ui_type_id,
            "ui_mask": ui_mask,
        }