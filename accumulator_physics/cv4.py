import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

f = np.load("feats.npz")
W, I, In, y = f["W"], f["I"], f["In"], f["y"]
cool = y[:, 0]
labels = y[:, :4]
grp = np.cumsum(np.r_[True, np.any(labels[1:] != labels[:-1], 1)]) - 1
rng = np.random.default_rng(0); u = rng.permutation(np.unique(grp)); size = np.zeros(4); asg = {}
for gi in u:
    j = int(np.argmin(size)); asg[gi] = j; size[j] += (grp == gi).sum()
fid = np.array([asg[gi] for gi in grp])
rf = lambda: RandomForestClassifier(300, n_jobs=-1, random_state=0)


def blockcv(X, yt):
    p = np.empty_like(yt)
    for j in range(4):
        tr, te = fid != j, fid == j
        p[te] = rf().fit(X[tr], yt[tr]).predict(X[te])
    return accuracy_score(yt, p)


def coolout(X, yt):
    return [accuracy_score(yt[cool == c], rf().fit(X[cool != c], yt[cool != c]).predict(X[cool == c])) for c in [3, 20, 100]]


t = np.arange(len(y))[:, None].astype(float)
timing = np.array([n.split("|")[-1] in ("t10", "t90", "rise") for n in In])
for nm, ti in [("valve", 1), ("pump", 2), ("accumulator", 3)]:
    yt = y[:, ti]
    print(f"{nm:12s} cycle-index-only blockCV {blockcv(t, yt):.3f}")
    for fn, X in [("whole", W), ("interval", I), ("timing-only", I[:, timing])]:
        c = coolout(X, yt)
        print(f"   {fn:12s} leave-one-cooler-out: c3 {c[0]:.3f} c20 {c[1]:.3f} c100 {c[2]:.3f}  mean {np.mean(c):.3f}")
