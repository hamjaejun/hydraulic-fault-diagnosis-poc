"""Level 1-3 evaluation ladder for all four components (run from 데이터/중간산출물).
L1  random 5-fold (stratified)
L2  block-aware 5-fold (blocks = runs of identical 4-label profile)
L3a combination holdout: 5-fold over the 144 (cooler,valve,pump,acc) combinations; every test combo unseen
L3b compound-fault generalisation: train on cycles with <=1 faulty component, test on cycles with >=2
L3c train on <=2 faulty components, test on 3-4
Each split also reports a 'cycle index only' model to expose measurement-order leakage.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

f = np.load("feats.npz"); g = np.load("phys.npz"); h = np.load("p0.npz")
W, I, In, y = f["W"], f["I"], f["In"], f["y"]
Kn = g["Kn"]; K = g["K"][:, [i for i, n in enumerate(Kn) if n.startswith("K|PS1")]]
p0T = np.nan_to_num(h["p0_n1.4_Qnom_T"], nan=np.nanmedian(h["p0_n1.4_Qnom_T"]))
p020 = np.nan_to_num(h["p0_n1.4_Qnom_20C"], nan=np.nanmedian(h["p0_n1.4_Qnom_20C"]))
timing = np.array([n.split("|")[-1] in ("t10", "t90", "rise") for n in In])
PHYS = np.c_[p0T, p020, h["T"], I[:, timing], K]
idx = np.arange(len(y))[:, None].astype(float)

lab = y[:, :4]
healthy_val = np.array([100, 100, 0, 130])
nfault = (lab != healthy_val).sum(1)
combo = np.unique(lab, axis=0, return_inverse=True)[1].ravel()
block = np.cumsum(np.r_[True, np.any(lab[1:] != lab[:-1], 1)]) - 1
print("cycles by number of faulty components:", {k: int((nfault == k).sum()) for k in range(5)})
print("combinations:", combo.max() + 1, " blocks:", block.max() + 1)


def group_folds(grp, k=5, seed=0):
    rng = np.random.default_rng(seed); u = rng.permutation(np.unique(grp)); size = np.zeros(k); asg = {}
    for gi in u:
        j = int(np.argmin(size)); asg[gi] = j; size[j] += (grp == gi).sum()
    fid = np.array([asg[gi] for gi in grp])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(k)]


def splits(name, yt):
    if name == "L1 random":
        return list(StratifiedKFold(5, shuffle=True, random_state=0).split(W, yt))
    if name == "L2 block":
        return group_folds(block)
    if name == "L3a combo-out":
        return group_folds(combo)
    if name == "L3b single->multi":
        return [(np.flatnonzero(nfault <= 1), np.flatnonzero(nfault >= 2))]
    if name == "L3c <=2 -> 3-4":
        return [(np.flatnonzero(nfault <= 2), np.flatnonzero(nfault >= 3))]


def run(X, yt, sp):
    P = np.full(len(yt), -1.0); used = np.zeros(len(yt), bool)
    for tr, te in sp:
        P[te] = RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(X[tr], yt[tr]).predict(X[te]); used[te] = True
    return accuracy_score(yt[used], P[used]), f1_score(yt[used], P[used], average="macro")


comps = [("cooler", 0), ("valve", 1), ("pump", 2), ("accumulator", 3)]
levels = ["L1 random", "L2 block", "L3a combo-out", "L3b single->multi", "L3c <=2 -> 3-4"]
for cn, ci in comps:
    yt = y[:, ci]
    print(f"\n=== {cn} (accuracy / macro-F1) ===")
    print(f"{'split':20s} {'cycle-index only':>18s} {'whole stats(187)':>18s} {'interval(1184)':>18s}" + (f" {'physics feats':>18s}" if cn == "accumulator" else ""))
    for lv in levels:
        sp = splits(lv, yt)
        cols = [run(idx, yt, sp), run(W, yt, sp), run(I, yt, sp)]
        if cn == "accumulator": cols.append(run(PHYS, yt, sp))
        print(f"{lv:20s} " + " ".join(f"{a:8.3f}/{b:.3f}    " for a, b in cols), flush=True)
