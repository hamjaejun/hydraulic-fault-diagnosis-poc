"""Experiment 12/13 (run from 데이터/중간산출물).
A. L4 = leave-one-cooler-out (unseen oil temperature) for valve, pump, accumulator.
B. Two-state calibration: at the unseen temperature we only measure k cycles with the accumulator at
   130 bar and k cycles at 90 bar (the two ends). Can we classify the never-seen 100/115 bar states there?
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

f = np.load("feats.npz"); g = np.load("phys.npz"); h = np.load("p0.npz")
W, I, In, y = f["W"], f["I"], f["In"], f["y"]
Kn = g["Kn"]; K = g["K"][:, [i for i, n in enumerate(Kn) if n.startswith("K|PS1")]]
T = h["T"]
p0T = np.nan_to_num(h["p0_n1.4_Qnom_T"], nan=np.nanmedian(h["p0_n1.4_Qnom_T"]))
p020 = np.nan_to_num(h["p0_n1.4_Qnom_20C"], nan=np.nanmedian(h["p0_n1.4_Qnom_20C"]))
timing = np.array([n.split("|")[-1] in ("t10", "t90", "rise") for n in In])
PHYS = np.c_[p0T, p020, T, I[:, timing], K]
idx = np.arange(len(y))[:, None].astype(float)
cool, ya = y[:, 0], y[:, 3]
classes = np.array([90, 100, 115, 130])
nearest = lambda v: classes[np.argmin(np.abs(v[:, None] - classes[None]), 1)]
rf = lambda s=0: RandomForestClassifier(200, n_jobs=-1, random_state=s)

print("== A. L4 leave-one-cooler-out: accuracy per held-out cooler (c3 / c20 / c100) and mean ==")
for cn, ci in [("valve", 1), ("pump", 2), ("accumulator", 3)]:
    yt = y[:, ci]
    sets = [("cycle-index", idx), ("whole(187)", W), ("interval(1184)", I)] + ([("physics", PHYS)] if ci == 3 else [])
    for nm, X in sets:
        r = [accuracy_score(yt[cool == c], rf().fit(X[cool != c], yt[cool != c]).predict(X[cool == c])) for c in [3, 20, 100]]
        print(f"  {cn:11s} {nm:15s} " + " / ".join(f"{v:.3f}" for v in r) + f"   mean {np.mean(r):.3f}", flush=True)

print("\n== B. two-state calibration at the unseen temperature (k cycles at 130 bar + k at 90 bar) ==")
print("   scored on target cycles not used for calibration; 'mid' = 100/115 bar only (never seen at target)")
ks = [1, 2, 5, 10]
methods = ["physics, source-calibrated (k=0)", "physics, 2-point target-only", "physics, source + target anchors",
           "ML interval, source + target anchors", "ML physics-feats, source + target anchors"]
res = {m: {k: [] for k in ks} for m in methods}
for c in [3, 20, 100]:
    src = np.flatnonzero(cool != c); tgt = np.flatnonzero(cool == c)
    for k in ks:
        for r in range(20):
            rng = np.random.default_rng(r)
            anc = np.r_[rng.choice(tgt[ya[tgt] == 130], k, replace=False), rng.choice(tgt[ya[tgt] == 90], k, replace=False)]
            te = np.setdiff1d(tgt, anc); mid = np.isin(ya[te], [100, 115])
            preds = {
                methods[0]: nearest(LinearRegression().fit(p020[src, None], ya[src]).predict(p020[te, None])),
                methods[1]: nearest(LinearRegression().fit(p020[anc, None], ya[anc]).predict(p020[te, None])),
                methods[2]: nearest(LinearRegression().fit(p020[np.r_[src, anc], None], ya[np.r_[src, anc]],
                                    sample_weight=np.r_[np.ones(len(src)), np.full(len(anc), len(src) / len(anc) / 2)]).predict(p020[te, None])),
            }
            if r < 3:  # RF is slow; fewer repeats
                for m, X in [(methods[3], I), (methods[4], PHYS)]:
                    tr = np.r_[src, anc]
                    preds[m] = rf(r).fit(X[tr], ya[tr]).predict(X[te])
            for m, p in preds.items():
                res[m][k].append((c, accuracy_score(ya[te], p), accuracy_score(ya[te][mid], p[mid])))
print(f"  {'method':42s}" + "".join(f"{'k=' + str(k) + ' all/mid':>16s}" for k in ks))
for m in methods:
    row = ""
    for k in ks:
        a = np.array([[v[1], v[2]] for v in res[m][k]]).mean(0)
        row += f"{a[0]:9.3f}/{a[1]:.3f} "
    print(f"  {m:42s}{row}")
print("\n  per target temperature at k=5 (all-class accuracy c3 / c20 / c100):")
for m in methods:
    per = [np.mean([v[1] for v in res[m][5] if v[0] == c]) for c in [3, 20, 100]]
    print(f"  {m:42s}" + " / ".join(f"{v:.3f}" for v in per))
