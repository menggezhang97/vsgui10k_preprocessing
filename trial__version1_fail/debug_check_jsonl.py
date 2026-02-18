import gzip, json, random, math
from collections import Counter

PATH = "trials_with_elements.jsonl.gz"
N_SAMPLE = 300        # 抽样检查 trial 数，想更严就调大
EPS = 1e-3            # 浮点误差容忍

random.seed(7)

def almost(a, b, eps=EPS):
    return abs(a - b) <= eps

def check_trial(tr):
    errs = []

    r_list = tr["params"]["r_list"]
    n_r = len(r_list)
    W = tr["meta"]["seg_shape"]["w"]
    H = tr["meta"]["seg_shape"]["h"]
    scan = tr["scanpath"]
    if not scan:
        return ["empty_scanpath"]

    # 1) time monotonic + dt consistency
    last_t = None
    for i, fx in enumerate(scan):
        t = fx["t"]
        dur = fx["dur"]
        dt = fx["dt"]
        t_end = fx["t_end"]

        if t < -EPS:
            errs.append(f"t_negative@{i}:{t}")

        if last_t is not None:
            if t + EPS < last_t:
                errs.append(f"t_not_monotonic@{i}:{last_t}->{t}")
            # dt should equal t-last_t (within tolerance) unless dt is None
            if dt is None:
                errs.append(f"dt_none_mid@{i}")
            else:
                if not almost(dt, t - last_t, eps=1e-4):
                    errs.append(f"dt_mismatch@{i}:{dt} vs {t-last_t}")
        else:
            # first dt should be None
            if dt is not None:
                errs.append(f"dt_first_not_none:{dt}")

        # t_end = t + dur
        if not almost(t_end, t + dur, eps=1e-4):
            errs.append(f"t_end_mismatch@{i}:{t_end} vs {t+dur}")

        last_t = t

    # 2) xy mapping check
    for i, fx in enumerate(scan[:50]):  # 前50个就够抓错
        x = fx["xy_norm"]["x"]
        y = fx["xy_norm"]["y"]
        xs = fx["xy_seg"]["x"]
        ys = fx["xy_seg"]["y"]
        if not almost(xs, x * W, eps=1e-2):
            errs.append(f"xy_seg_x_mismatch@{i}:{xs} vs {x*W}")
        if not almost(ys, y * H, eps=1e-2):
            errs.append(f"xy_seg_y_mismatch@{i}:{ys} vs {y*H}")

    # 3) in_target vector format
    for i, fx in enumerate(scan[:50]):
        it = fx["hit"]["in_target"]
        if not isinstance(it, list):
            errs.append(f"in_target_not_list@{i}")
            break
        if len(it) != n_r:
            errs.append(f"in_target_len_mismatch@{i}:{len(it)} vs {n_r}")
            break

    # 4) dist==0 implies r1 hit (if target exists)
    tgt_bbox = tr["target"]["bbox_seg"]
    if tgt_bbox is not None:
        for i, fx in enumerate(scan[:100]):
            dist = fx["hit"]["dist_to_target"]
            it = fx["hit"]["in_target"]
            if dist is not None and dist <= 1e-6:
                # r1 is index 0 because r_list starts with 1.0
                if it[0] != 1:
                    errs.append(f"dist0_but_r1_miss@{i}")

    # 5) summary consistency for one r (primary r)
    primary_r = tr["params"]["primary_r_for_found"]
    try:
        j = r_list.index(primary_r)
    except ValueError:
        j = 0
    hits = [fx["hit"]["in_target"][j] for fx in scan]
    sum_any = tr["summary"]["success"]["any"]
    sum_dwell = tr["summary"]["success"]["dwell"]
    # keys are r1, r1_5, r2...
    def r_key(r):
        if abs(r - round(r)) < 1e-9:
            return f"r{int(round(r))}"
        return f"r{str(r).replace('.','_')}"
    rk = r_key(primary_r)

    if rk in sum_any:
        if int(any(hits)) != int(sum_any[rk]):
            errs.append(f"summary_any_mismatch@{rk}")
    if rk in sum_dwell:
        if int(sum(hits) >= tr["params"]["dwell_min_hits"]) != int(sum_dwell[rk]):
            errs.append(f"summary_dwell_mismatch@{rk}")

    # found consistency
    found = tr["summary"]["found"]
    idx_found = found["idx_found"]
    if idx_found is None:
        # then dwell should be 0 for primary r
        if int(sum(hits) >= tr["params"]["dwell_min_hits"]) == 1:
            errs.append("found_none_but_dwell_true")
    else:
        if not (0 <= idx_found < len(scan)):
            errs.append("idx_found_out_of_range")
        else:
            # at idx_found, cumulative hits >= dwell_min_hits should hold
            m = tr["params"]["dwell_min_hits"]
            if sum(hits[:idx_found+1]) < m:
                errs.append("idx_found_not_meeting_dwell")
            # t_found should equal scan[idx_found]["t"]
            t_found = found["t_found"]
            if t_found is None or not almost(t_found, scan[idx_found]["t"], eps=1e-6):
                errs.append("t_found_mismatch")

    return errs


def main():
    # reservoir sample trials
    sample = []
    n = 0
    with gzip.open(PATH, "rt", encoding="utf-8") as f:
        for line in f:
            tr = json.loads(line)
            n += 1
            if len(sample) < N_SAMPLE:
                sample.append(tr)
            else:
                j = random.randint(0, n-1)
                if j < N_SAMPLE:
                    sample[j] = tr

    print("Total trials read (stream):", n)
    print("Sampled trials:", len(sample))

    err_counter = Counter()
    bad_trials = 0

    for tr in sample:
        errs = check_trial(tr)
        if errs:
            bad_trials += 1
            for e in errs:
                err_counter[e] += 1

    print("Bad trials:", bad_trials, "/", len(sample))
    if err_counter:
        print("\nTop errors:")
        for k, v in err_counter.most_common(20):
            print(f"{v:4d}  {k}")
    else:
        print("No issues found in sampled trials.")

if __name__ == "__main__":
    main()
