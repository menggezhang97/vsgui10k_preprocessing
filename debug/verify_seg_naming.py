import os
import glob
import json
import random
import pandas as pd

# ---------- base paths (run from C:\Users\zhang\GUI\VSGUI10K\data) ----------
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "data")
FIX_CSV  = os.path.join(DATA_DIR, "vsgui10k_fixations.csv")
IMG_DIR  = os.path.join(DATA_DIR, "vsgui10k-images")
SEG_ROOT = os.path.join(DATA_DIR, "segmentation")

SAMPLE_N = 200  # sample trials for checking
random.seed(7)


def build_seg_index(seg_root: str) -> dict:
    idx = {}
    pattern = os.path.join(seg_root, "block *", "*.json")
    for p in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(p))[0]
        idx.setdefault(stem, p)
    if not idx:
        for p in glob.glob(os.path.join(seg_root, "**", "*.json"), recursive=True):
            stem = os.path.splitext(os.path.basename(p))[0]
            idx.setdefault(stem, p)
    return idx


def file_exists_in_images(img_dir, fname):
    return os.path.exists(os.path.join(img_dir, fname))


def seg_exists(seg_idx, stem):
    return stem in seg_idx and os.path.exists(seg_idx[stem])


def main():
    print("BASE:", BASE)
    print("FIX_CSV:", FIX_CSV)
    print("IMG_DIR:", IMG_DIR)
    print("SEG_ROOT:", SEG_ROOT)
    print()

    assert os.path.exists(FIX_CSV), f"Missing {FIX_CSV}"
    assert os.path.isdir(IMG_DIR), f"Missing {IMG_DIR}"
    assert os.path.isdir(SEG_ROOT), f"Missing {SEG_ROOT}"

    seg_idx = build_seg_index(SEG_ROOT)
    print("Seg index size:", len(seg_idx))

    df = pd.read_csv(FIX_CSV, low_memory=False)

    # we only care about img_type==2 for the official search-phase pipeline
    df2 = df[df["img_type"] == 2].copy()

    # trial key at least includes these
    key = ["pid", "img_name", "new_img_name", "tgt_id", "cue", "absent"]
    df2 = df2[key].drop_duplicates()

    print("Unique search-phase trial keys:", len(df2))

    if len(df2) == 0:
        print("No img_type==2 rows found. Stop.")
        return

    # sample
    sample = df2.sample(min(SAMPLE_N, len(df2)), random_state=7)

    cnt = {
        "img_name_seg_hit": 0,
        "new_img_name_seg_hit": 0,
        "img_name_imgfile_hit": 0,
        "new_img_name_imgfile_hit": 0,
        "both_seg_hit": 0,
        "neither_seg_hit": 0,
        "both_imgfile_hit": 0,
        "neither_imgfile_hit": 0,
    }

    examples = {
        "seg_img_name_only": [],
        "seg_new_img_only": [],
        "seg_neither": [],
        "imgfile_img_name_only": [],
        "imgfile_new_img_only": [],
        "imgfile_neither": [],
    }

    for _, r in sample.iterrows():
        img_name = str(r["img_name"])
        new_img_name = str(r["new_img_name"])

        stem_img = os.path.splitext(img_name)[0]
        stem_new = os.path.splitext(new_img_name)[0]

        seg_img = seg_exists(seg_idx, stem_img)
        seg_new = seg_exists(seg_idx, stem_new)

        imgfile_img = file_exists_in_images(IMG_DIR, img_name)
        imgfile_new = file_exists_in_images(IMG_DIR, new_img_name)

        if seg_img:
            cnt["img_name_seg_hit"] += 1
        if seg_new:
            cnt["new_img_name_seg_hit"] += 1
        if seg_img and seg_new:
            cnt["both_seg_hit"] += 1
        if (not seg_img) and (not seg_new):
            cnt["neither_seg_hit"] += 1

        if imgfile_img:
            cnt["img_name_imgfile_hit"] += 1
        if imgfile_new:
            cnt["new_img_name_imgfile_hit"] += 1
        if imgfile_img and imgfile_new:
            cnt["both_imgfile_hit"] += 1
        if (not imgfile_img) and (not imgfile_new):
            cnt["neither_imgfile_hit"] += 1

        # keep a few examples
        if seg_img and (not seg_new) and len(examples["seg_img_name_only"]) < 5:
            examples["seg_img_name_only"].append((img_name, new_img_name, stem_img, stem_new))
        if seg_new and (not seg_img) and len(examples["seg_new_img_only"]) < 5:
            examples["seg_new_img_only"].append((img_name, new_img_name, stem_img, stem_new))
        if (not seg_img) and (not seg_new) and len(examples["seg_neither"]) < 5:
            examples["seg_neither"].append((img_name, new_img_name, stem_img, stem_new))

        if imgfile_img and (not imgfile_new) and len(examples["imgfile_img_name_only"]) < 5:
            examples["imgfile_img_name_only"].append((img_name, new_img_name))
        if imgfile_new and (not imgfile_img) and len(examples["imgfile_new_img_only"]) < 5:
            examples["imgfile_new_img_only"].append((img_name, new_img_name))
        if (not imgfile_img) and (not imgfile_new) and len(examples["imgfile_neither"]) < 5:
            examples["imgfile_neither"].append((img_name, new_img_name))

    n = len(sample)
    print("\n=== SEGMENTATION NAME MATCH (sampled) ===")
    print(f"sample_n = {n}")
    print("seg hit by img_name stem     :", cnt["img_name_seg_hit"], f"({cnt['img_name_seg_hit']/n:.3f})")
    print("seg hit by new_img_name stem :", cnt["new_img_name_seg_hit"], f"({cnt['new_img_name_seg_hit']/n:.3f})")
    print("seg hit by BOTH              :", cnt["both_seg_hit"], f"({cnt['both_seg_hit']/n:.3f})")
    print("seg hit by NEITHER           :", cnt["neither_seg_hit"], f"({cnt['neither_seg_hit']/n:.3f})")

    print("\n=== IMAGE FILE EXISTS IN vsgui10k-images (sampled) ===")
    print("img file exists by img_name     :", cnt["img_name_imgfile_hit"], f"({cnt['img_name_imgfile_hit']/n:.3f})")
    print("img file exists by new_img_name :", cnt["new_img_name_imgfile_hit"], f"({cnt['new_img_name_imgfile_hit']/n:.3f})")
    print("img exists by BOTH              :", cnt["both_imgfile_hit"], f"({cnt['both_imgfile_hit']/n:.3f})")
    print("img exists by NEITHER           :", cnt["neither_imgfile_hit"], f"({cnt['neither_imgfile_hit']/n:.3f})")

    print("\n=== EXAMPLES ===")
    print("seg matches img_name only:", examples["seg_img_name_only"])
    print("seg matches new_img_name only:", examples["seg_new_img_only"])
    print("seg matches neither:", examples["seg_neither"])
    print("imgfile matches img_name only:", examples["imgfile_img_name_only"])
    print("imgfile matches new_img_name only:", examples["imgfile_new_img_only"])
    print("imgfile matches neither:", examples["imgfile_neither"])

    # Decision recommendation
    print("\n=== RECOMMENDATION (based on sample) ===")
    if cnt["img_name_seg_hit"] > cnt["new_img_name_seg_hit"]:
        print("Use seg key = stem(img_name)")
    elif cnt["new_img_name_seg_hit"] > cnt["img_name_seg_hit"]:
        print("Use seg key = stem(new_img_name)")
    else:
        print("Seg key ambiguous; need to inspect seg folder naming directly (list stems).")


if __name__ == "__main__":
    main()
