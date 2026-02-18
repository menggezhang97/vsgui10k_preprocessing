# save as quick_time_audit.py and run: py .\quick_time_audit.py
import pandas as pd
import numpy as np

FIX="vsgui10k_fixations.csv"
TRIAL_KEY=["pid","img_name","tgt_id","cue","absent"]

df=pd.read_csv(FIX)
df=df[(df["FPOGV"]==1)&(df["BPOGV"]==1)]
df=df.sort_values(TRIAL_KEY+["TIME"])

g=df.groupby(TRIAL_KEY, sort=False)

rows=[]
for k,sub in list(g)[:30]:
    tmin=float(sub["TIME"].min())
    tmax=float(sub["TIME"].max())
    rows.append((k[0],k[1],k[2],k[3],int(k[4]),len(sub),tmin,tmax,tmax-tmin))
print("sample 30 trials: pid,img,tgt,cue,abs,n_fix,tmin,tmax,dur")
for r in rows:
    print(r)

dur=[r[-1] for r in rows]
print("\nDur stats:", np.min(dur), np.median(dur), np.max(dur))
