import os
import pandas as pd

FILES = {
    "fixations": "vsgui10k_fixations.csv",
    "targets": "vsgui10k_targets.csv",
    "aim": "vsgui10k_aim_results.csv",
}

N_HEAD = 8          # head 行数
N_UNIQUE_SHOW = 12  # 每列展示多少 unique 值（用于找 phase/screen 字段）


def load_csv(path):
    # robust reading
    return pd.read_csv(path, low_memory=False)


def print_basic(df, name):
    print(f"\n==================== {name} ====================")
    print("rows:", len(df), "cols:", len(df.columns))
    print("\nCOLUMNS:")
    for c in df.columns:
        miss = df[c].isna().mean()
        dtype = str(df[c].dtype)
        print(f"  {c:30s}  dtype={dtype:10s}  missing={miss:.3f}")
    print("\nHEAD:")
    print(df.head(N_HEAD).to_string(index=False))


def guess_key_columns(df):
    # common candidate keys for joining
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if cl in ("pid", "participant", "participant_id", "subject", "subj"):
            candidates.append(c)
        elif "media" in cl and "id" in cl:
            candidates.append(c)
        elif cl in ("img_name", "image", "image_name", "stimulus", "stimulus_name", "filename"):
            candidates.append(c)
        elif "trial" in cl and "id" in cl:
            candidates.append(c)
        elif cl in ("tgt_id", "target_id", "target"):
            candidates.append(c)
        elif cl in ("cue", "prompt", "query"):
            candidates.append(c)
        elif cl in ("absent", "target_absent"):
            candidates.append(c)
    # keep order but unique
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def find_phase_like_columns(df):
    # columns that may encode screen phase / event types
    phase_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in [
            "phase", "screen", "event", "segment", "message",
            "stim", "display", "condition", "block", "trialtype"
        ]):
            phase_cols.append(c)
    return phase_cols


def show_uniques(df, cols, name):
    if not cols:
        print(f"\nNo phase-like / event-like columns detected in {name}.")
        return
    print(f"\nPotential phase/screen/event columns in {name}:")
    for c in cols:
        ser = df[c]
        # skip super-high-cardinality numeric columns
        nunq = ser.nunique(dropna=True)
        if nunq == 0:
            continue
        print(f"\n  - {c} (n_unique={nunq})")
        # show a few unique values
        vals = ser.dropna().unique()
        vals = vals[:N_UNIQUE_SHOW]
        for v in vals:
            s = str(v)
            if len(s) > 140:
                s = s[:140] + "..."
            print("     ", s)


def overlap_keys(dfs):
    # find column name overlaps across dfs
    names = list(dfs.keys())
    sets = {k: set(dfs[k].columns) for k in names}

    print("\n==================== COLUMN OVERLAPS ====================")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            inter = sorted(list(sets[a] & sets[b]))
            print(f"\nOverlap {a} ∩ {b}: {len(inter)} cols")
            if inter:
                print("  ", inter)


def try_join_report(df_fix, df_tgt, df_aim):
    print("\n==================== JOIN SANITY CHECK ====================")

    # candidate key sets by intersection
    fix_cols = set(df_fix.columns)
    tgt_cols = set(df_tgt.columns)
    aim_cols = set(df_aim.columns)

    common_fix_tgt = [c for c in ["pid", "media_id", "img_name", "tgt_id", "cue", "absent", "trial_id"]
                      if c in fix_cols and c in tgt_cols]
    common_fix_aim = [c for c in ["pid", "media_id", "img_name", "tgt_id", "cue", "absent", "trial_id"]
                      if c in fix_cols and c in aim_cols]
    common_tgt_aim = [c for c in ["pid", "media_id", "img_name", "tgt_id", "cue", "absent", "trial_id"]
                      if c in tgt_cols and c in aim_cols]

    print("Common candidate keys fix↔tgt:", common_fix_tgt)
    print("Common candidate keys fix↔aim:", common_fix_aim)
    print("Common candidate keys tgt↔aim:", common_tgt_aim)

    # attempt a lightweight merge on best-available keys
    def best_keys(keys):
        # prefer pid+media_id if available, then img_name, then tgt_id/cue/absent
        order = ["pid", "media_id", "img_name", "tgt_id", "cue", "absent", "trial_id"]
        out = [k for k in order if k in keys]
        return out[:5]  # cap

    keys_ft = best_keys(common_fix_tgt)
    keys_fa = best_keys(common_fix_aim)
    keys_ta = best_keys(common_tgt_aim)

    def merge_rate(a, b, keys, name):
        if not keys:
            print(f"\n{name}: no common keys to merge.")
            return
        # drop duplicates on keys to estimate join coverage
        a1 = a[keys].drop_duplicates()
        b1 = b[keys].drop_duplicates()
        m = a1.merge(b1, on=keys, how="inner")
        print(f"\n{name}: keys={keys}")
        print("  unique A:", len(a1), "unique B:", len(b1), "matched:", len(m))
        if len(a1) > 0:
            print("  match rate vs A:", round(len(m)/len(a1), 4))
        if len(b1) > 0:
            print("  match rate vs B:", round(len(m)/len(b1), 4))

    merge_rate(df_fix, df_tgt, keys_ft, "fix↔tgt")
    merge_rate(df_fix, df_aim, keys_fa, "fix↔aim")
    merge_rate(df_tgt, df_aim, keys_ta, "tgt↔aim")


def main():
    dfs = {}
    for name, fname in FILES.items():
        if not os.path.exists(fname):
            print(f"Missing file: {fname}")
            return
        dfs[name] = load_csv(fname)

    # Basic info
    for name in ["fixations", "targets", "aim"]:
        print_basic(dfs[name], name)

    # Key candidates
    print("\n==================== KEY CANDIDATES ====================")
    for name in ["fixations", "targets", "aim"]:
        keys = guess_key_columns(dfs[name])
        print(f"{name}: {keys}")

    # Phase/screen/event candidates
    print("\n==================== PHASE/EVENT CANDIDATES ====================")
    for name in ["fixations", "targets", "aim"]:
        cols = find_phase_like_columns(dfs[name])
        show_uniques(dfs[name], cols, name)

    # Column overlaps
    overlap_keys(dfs)

    # Join sanity
    try_join_report(dfs["fixations"], dfs["targets"], dfs["aim"])


if __name__ == "__main__":
    main()
