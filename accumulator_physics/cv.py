import numpy as np, json, sys
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

f = np.load("feats.npz")
W, I, y = f["W"], f["I"], f["y"]
labels = y[:, :4]
# Enejor-style blocks: run-length encoding of the 4 labels jointly
chg = np.r_[True, np.any(labels[1:] != labels[:-1], 1)]
grp = np.cumsum(chg) - 1
print("blocks:", grp.max() + 1)


def folds(seed, k=4):
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(grp)); size = np.zeros(k); asg = {}
    for gi in u:
        j = int(np.argmin(size)); asg[gi] = j; size[j] += (grp == gi).sum()
    fid = np.array([asg[gi] for gi in grp])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(k)]


def models():
    return {"LR": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
            "RF": RandomForestClassifier(300, n_jobs=-1, random_state=0),
            "GB": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)}


sets = {"whole(187)": W, "interval(1184)": I, "whole+interval": np.hstack([W, I])}
targets = {"valve": 1, "accumulator": 3}
seeds = [0, 1, 2]
res = {}
for tn, ti in targets.items():
    yt = y[:, ti]
    for sn, X in sets.items():
        for mn in ["LR", "RF", "GB"]:
            acc, f1 = [], []
            for s in seeds:
                for tr, te in folds(s):
                    m = models()[mn].fit(X[tr], yt[tr]); p = m.predict(X[te])
                    acc.append(accuracy_score(yt[te], p)); f1.append(f1_score(yt[te], p, average="macro"))
            res[f"{tn}|{sn}|{mn}"] = (float(np.mean(acc)), float(np.std(acc)), float(np.mean(f1)))
            print(f"{tn:12s} {sn:16s} {mn}: acc {np.mean(acc):.3f}±{np.std(acc):.3f}  mF1 {np.mean(f1):.3f}", flush=True)
json.dump(res, open("cv_results.json", "w"), indent=1)
