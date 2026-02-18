import os
import json
import numpy as np
import matplotlib.pyplot as plt

JSONL = "trials_with_elements.jsonl"
OUT_DIR = "sanity_plots"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    trial_hit_ratios = []
    trial_bg_ratios = []
    nfix_list = []

    total_fix = 0
    total_bg = 0
    total_hit = 0

    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            scan = t["scanpath"]
            if not scan:
                continue
            n = len(scan)
            nfix_list.append(n)

            bg = sum(1 for s in scan if s["elem_class"] == "Background")
            hit = n - bg

            total_fix += n
            total_bg += bg
            total_hit += hit

            trial_hit_ratios.append(hit / n)
            trial_bg_ratios.append(bg / n)

    print("Total fix:", total_fix)
    print("Overall hit ratio:", round(total_hit / total_fix, 4))
    print("Overall background ratio:", round(total_bg / total_fix, 4))
    print("Trials:", len(trial_hit_ratios))

    # 1) Histogram: trial-level hit ratio
    plt.figure()
    plt.hist(trial_hit_ratios, bins=30)
    plt.title("Per-trial Hit Ratio Distribution (1 - Background)")
    plt.xlabel("Hit ratio")
    plt.ylabel("Count")
    out1 = os.path.join(OUT_DIR, "hist_trial_hit_ratio.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    # 2) Histogram: trial-level background ratio
    plt.figure()
    plt.hist(trial_bg_ratios, bins=30)
    plt.title("Per-trial Background Ratio Distribution")
    plt.xlabel("Background ratio")
    plt.ylabel("Count")
    out2 = os.path.join(OUT_DIR, "hist_trial_background_ratio.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    # 3) Histogram: n_fix per trial
    plt.figure()
    plt.hist(nfix_list, bins=30)
    plt.title("Fixations per Trial Distribution")
    plt.xlabel("n_fix")
    plt.ylabel("Count")
    out3 = os.path.join(OUT_DIR, "hist_nfix_per_trial.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    plt.close()

    print("Saved plots to:", OUT_DIR)
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)

if __name__ == "__main__":
    main()
