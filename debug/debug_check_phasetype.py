import pandas as pd

df = pd.read_csv("vsgui10k_fixations.csv")

print("img_type distribution:")
print(df["img_type"].value_counts())

print("\nExample trial phases:")
sample = df[df["pid"]=="015f21"]
print(sample.groupby(["img_name","tgt_id","cue","absent"])["img_type"].unique().head(10))
