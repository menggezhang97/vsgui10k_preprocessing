import os, json, gzip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

JSONL = "trials_with_elements.jsonl.gz"
IMG_DIR = r"data\vsgui10k-images"

# 这张你截图里显示的是 015f21-63 0ff0be.png
PID = "015f21"
IMG = "0ff0be.png"
MEDIA_ID = 63

SCREEN_W = 1920
SCREEN_H = 1200

OUT_A = "debug_compare_A_current.png"
OUT_B = "debug_compare_B_screen2stim.png"

def find_trial():
    with gzip.open(JSONL, "rt", encoding="utf-8") as f:
        for line in f:
            tr = json.loads(line)
            if tr["key"]["pid"] == PID and tr["key"]["img_name"] == IMG and tr["meta"]["media_id"] == MEDIA_ID:
                return tr
    return None

def plot_overlay(img_path, W_show, H_show, xs, ys, title, out_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_show, H_show), Image.BILINEAR)

    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.title(title)

    # 连接线 + 编号（更容易看错位）
    plt.plot(xs, ys, linewidth=1)
    plt.scatter([xs[0]],[ys[0]], s=160, marker="s")   # start
    plt.scatter([xs[-1]],[ys[-1]], s=180, marker="^") # end

    show_idx = list(range(min(10, len(xs)))) + list(range(max(0, len(xs)-5), len(xs)))
    show_idx = sorted(set(show_idx))
    for i in show_idx:
        plt.text(xs[i]+3, ys[i]+3, str(i), fontsize=8)

    plt.xlim([0, W_show])
    plt.ylim([H_show, 0])
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def main():
    tr = find_trial()
    if tr is None:
        print("Trial not found.")
        return

    img_path = os.path.join(IMG_DIR, tr["key"]["img_name"])
    scan = tr["scanpath"]
    W_seg = tr["meta"]["seg_shape"]["w"]
    H_seg = tr["meta"]["seg_shape"]["h"]

    # -------------------------
    # A) 你当前方法：当作 stimulus-normalized
    # -------------------------
    xs_A = [fx["xy_seg"]["x"] for fx in scan]
    ys_A = [fx["xy_seg"]["y"] for fx in scan]
    plot_overlay(
        img_path, W_seg, H_seg,
        xs_A, ys_A,
        "A) Current mapping: FPOGX/FPOGY treated as stimulus-normalized",
        OUT_A
    )

    # -------------------------
    # B) 屏幕->刺激反推（关键）
    # 假设 FPOGX/FPOGY 是 screen-normalized
    # 刺激在屏幕上居中显示：offset = (screen - disp)/2
    # disp_W = original_width * scale_x, disp_H = original_height * scale_y
    # gaze_screen_px = (FPOGX*SCREEN_W, FPOGY*SCREEN_H)
    # gaze_stim_norm = (gaze_screen_px - offset)/disp_size
    # 再映射到 seg
    # -------------------------
    tgt = tr["target"]
    orig_W = tgt.get("original_width", None)
    orig_H = tgt.get("original_height", None)
    sx = tgt.get("scale_x", None)
    sy = tgt.get("scale_y", None)

    # 如果缺失就没法做这个对比
    if orig_W is None or orig_H is None or sx is None or sy is None:
        print("Missing original_width/original_height/scale_x/scale_y for this trial.")
        return

    disp_W = float(orig_W) * float(sx)
    disp_H = float(orig_H) * float(sy)
    off_x = (SCREEN_W - disp_W) * 0.5
    off_y = (SCREEN_H - disp_H) * 0.5

    xs_B = []
    ys_B = []
    for fx in scan:
        gx = float(fx["xy_norm_raw"]["x"])  # 这里存的是原始 FPOGX
        gy = float(fx["xy_norm_raw"]["y"])  # 原始 FPOGY

        # screen px
        sx_px = gx * SCREEN_W
        sy_px = gy * SCREEN_H

        # back to stimulus normalized
        stim_x = (sx_px - off_x) / disp_W
        stim_y = (sy_px - off_y) / disp_H

        # clip
        stim_x = max(0.0, min(1.0, stim_x))
        stim_y = max(0.0, min(1.0, stim_y))

        xs_B.append(stim_x * W_seg)
        ys_B.append(stim_y * H_seg)

    plot_overlay(
        img_path, W_seg, H_seg,
        xs_B, ys_B,
        "B) Screen->Stimulus mapping: assume FPOGX/FPOGY are screen-normalized + centered stimulus",
        OUT_B
    )

    print("Saved:", OUT_A)
    print("Saved:", OUT_B)
    print("Compare the two overlays visually.")

if __name__ == "__main__":
    main()
