"""Two-state 'adapter': keep the source curve shape, re-map the target's p0 estimate with 2 parameters (scale, offset)
so that the target 90/130-bar anchors land where the source 90/130 bar sit. Then classify with the source curve."""
import numpy as np
from sklearn.metrics import accuracy_score
f = np.load("feats.npz"); h = np.load("p0.npz"); y = f["y"]
p = np.nan_to_num(h["p0_n1.4_Qnom_20C"], nan=np.nanmedian(h["p0_n1.4_Qnom_20C"]))
cool, ya = y[:, 0], y[:, 3]; classes = np.array([90, 100, 115, 130])
for nm in ["nearest source class-median", "source quadratic curve"]:
    out = {k: [] for k in [1, 2, 5, 10]}
    for c in [3, 20, 100]:
        src = cool != c; tgt = np.flatnonzero(cool == c)
        med = np.array([np.median(p[src & (ya == a)]) for a in classes])
        coef = np.polyfit(p[src], ya[src], 2)
        for k in out:
            for r in range(50):
                rng = np.random.default_rng(r)
                a1 = rng.choice(tgt[ya[tgt] == 90], k, replace=False); a2 = rng.choice(tgt[ya[tgt] == 130], k, replace=False)
                te = np.setdiff1d(tgt, np.r_[a1, a2]); mid = np.isin(ya[te], [100, 115])
                s = (med[3] - med[0]) / (np.median(p[a2]) - np.median(p[a1])); b = med[0] - s * np.median(p[a1])
                q = s * p[te] + b
                if nm.startswith("nearest"):
                    pr = classes[np.argmin(np.abs(q[:, None] - med[None]), 1)]
                else:
                    v = np.polyval(coef, q); pr = classes[np.argmin(np.abs(v[:, None] - classes[None]), 1)]
                out[k].append((c, accuracy_score(ya[te], pr), accuracy_score(ya[te][mid], pr[mid])))
    print(nm)
    for k, v in out.items():
        a = np.array([[x[1], x[2]] for x in v]).mean(0)
        per = [np.mean([x[1] for x in v if x[0] == c]) for c in [3, 20, 100]]
        print(f"   k={k:2d}  all {a[0]:.3f}  mid {a[1]:.3f}   per c3/c20/c100 " + " / ".join(f"{u:.3f}" for u in per))
