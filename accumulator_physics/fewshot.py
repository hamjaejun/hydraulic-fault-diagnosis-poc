"""Experiment 8: is physics+calibration data-efficient?
A. Learning curve: N labelled training cycles drawn from the two source temperatures, test on the unseen one.
B. Normal-only adaptation: no fault labels from the target temperature, only k 'healthy' cycles
   (accumulator 130 bar, pump no leak, valve 100 %) used to shift features (adapter = offset only).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

f = np.load("feats.npz"); g = np.load("phys.npz"); h = np.load("p0.npz")
W, I, In, y = f["W"], f["I"], f["In"], f["y"]
Kn = g["Kn"]; K = g["K"][:, [i for i, n in enumerate(Kn) if n.startswith("K|PS1")]]
T = h["T"]
p020 = h["p0_n1.4_Qnom_20C"]; p020 = np.nan_to_num(p020, nan=np.nanmedian(p020))
p0T = h["p0_n1.4_Qnom_T"]; p0T = np.nan_to_num(p0T, nan=np.nanmedian(p0T))
timing = np.array([n.split("|")[-1] in ("t10", "t90", "rise") for n in In])
ya, cool = y[:, 3], y[:, 0]
classes = np.array([90, 100, 115, 130])
healthy = (y[:, 3] == 130) & (y[:, 2] == 0) & (y[:, 1] == 100)
nearest = lambda v: classes[np.argmin(np.abs(v[:, None] - classes[None]), 1)]

X_phys_ml = np.c_[p0T, p020, T, I[:, timing], K]
methods = {
    "pure ML: whole stats (187)": ("rf", W),
    "pure ML: interval (1184)": ("rf", I),
    "physics + linear calib": ("lin", p020[:, None]),
    "physics + ML": ("rf", X_phys_ml),
}


def fit_pred(kind, Xtr, ytr, Xte, seed):
    if kind == "rf":
        return RandomForestClassifier(200, n_jobs=-1, random_state=seed).fit(Xtr, ytr).predict(Xte)
    return nearest(LinearRegression().fit(Xtr, ytr).predict(Xte))


def draw(idx, n, rng):
    per = max(1, n // 4); out = []
    for a in classes:
        pool = idx[ya[idx] == a]
        out.append(rng.choice(pool, min(per, len(pool)), replace=False))
    return np.concatenate(out)


print("== A. learning curve: accuracy on unseen temperature (mean over 3 targets x 15 draws) ==")
Ns = [8, 20, 40, 100, 400, 1400]
print(f"{'method':30s}" + "".join(f"{'N=' + str(n):>9s}" for n in Ns))
for nm, (kind, X) in methods.items():
    row = []
    for n in Ns:
        accs = []
        for c in [3, 20, 100]:
            src = np.flatnonzero(cool != c); te = cool == c
            reps = 1 if n >= 1400 else 15
            for r in range(reps):
                rng = np.random.default_rng(r)
                tr = src if n >= 1400 else draw(src, n, rng)
                accs.append(accuracy_score(ya[te], fit_pred(kind, X[tr], ya[tr], X[te], r)))
        row.append(np.mean(accs))
    print(f"{nm:30s}" + "".join(f"{v:9.3f}" for v in row), flush=True)

print("\n== B. normal-only adaptation on the unseen temperature (all source labels, k healthy target cycles) ==")
print("   adapter = shift every feature by (mean of target healthy - mean of source healthy)")
ks = [0, 5, 20]
print(f"{'method':30s}" + "".join(f"{'k=' + str(k):>9s}" for k in ks) + "   (per target c3/c20/c100 at k=5)")
for nm, (kind, X) in methods.items():
    row, per5 = [], []
    for k in ks:
        accs = []
        for c in [3, 20, 100]:
            src = cool != c; te = np.flatnonzero(cool == c)
            reps = 1 if k == 0 else 10
            a_c = []
            for r in range(reps):
                Xa = X.astype(float).copy()
                if k > 0:
                    rng = np.random.default_rng(r)
                    ht = rng.choice(np.flatnonzero((cool == c) & healthy), k, replace=False)
                    shift = Xa[ht].mean(0) - Xa[src & healthy].mean(0)
                    Xa[te] = Xa[te] - shift
                    keep = np.setdiff1d(te, ht)
                else:
                    keep = te
                a_c.append(accuracy_score(ya[keep], fit_pred(kind, Xa[src], ya[src], Xa[keep], r)))
            accs.append(np.mean(a_c))
            if k == 5: per5.append(np.mean(a_c))
        row.append(np.mean(accs))
    print(f"{nm:30s}" + "".join(f"{v:9.3f}" for v in row) + "   " + " / ".join(f"{v:.3f}" for v in per5), flush=True)
