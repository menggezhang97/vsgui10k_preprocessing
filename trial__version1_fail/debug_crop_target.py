import gzip, json, os
from PIL import Image

JSONL = "trials_with_elements.jsonl.gz"
IMG_DIR = r"data\vsgui10k-images"

PID = "015f21"
IMG = "0ff0be.png"
MEDIA_ID = 63

OUT_CROP = "crop_target.png"
OUT_CTX  = "crop_context.png"

def main():
    trial = None
    with gzip.open(JSONL, "rt", encoding="utf-8") as f:
        for line in f:
            tr = json.loads(line)
            if tr["key"]["pid"] == PID and tr["key"]["img_name"] == IMG and tr["meta"]["media_id"] == MEDIA_ID:
                trial = tr
                break
    if trial is None:
        print("Trial not found")
        return

    bbox = trial["target"]["bbox_seg"]
    if bbox is None:
        print("No target bbox")
        return

    x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    img_path = os.path.join(IMG_DIR, IMG)
    im = Image.open(img_path).convert("RGB")

    W, H = trial["meta"]["seg_shape"]["w"], trial["meta"]["seg_shape"]["h"]
    im = im.resize((W, H), Image.BILINEAR)

    # tight crop
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    crop_big = crop.resize((crop.size[0]*6, crop.size[1]*6), Image.NEAREST)
    crop_big.save(OUT_CROP)

    # context crop (padding around)
    pad = 80
    cx0 = max(0, int(x0 - pad)); cy0 = max(0, int(y0 - pad))
    cx1 = min(W, int(x1 + pad)); cy1 = min(H, int(y1 + pad))
    ctx = im.crop((cx0, cy0, cx1, cy1))
    ctx_big = ctx.resize((ctx.size[0]*2, ctx.size[1]*2), Image.BILINEAR)
    ctx_big.save(OUT_CTX)

    print("Saved:", OUT_CROP)
    print("Saved:", OUT_CTX)
    print("Cue:", trial["task"]["cue"], "Absent:", trial["task"]["absent"])

if __name__ == "__main__":
    main()
