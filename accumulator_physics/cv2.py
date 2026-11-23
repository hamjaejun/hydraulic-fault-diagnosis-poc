import numpy as np, re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

f = np.load("feats.npz")
W, I, In, y = f["W"], f["I"], f["In"], f["y"]
ya, cool, stable = y[:, 3], y[:, 0], y[:, 4]
labels = y[:, :4]
grp = np.cumsum(np.r_[True, np.any(labels[1:] != labels[:-1], 1)]) - 1
# interval end times (s) for causal truncation
ends = [1.5, 9.2, 11.2, 18.8, 20.8, 28.9, 30.9, 38.9, 40.9, 48.8, 50.8, 60.0]
stepend = [1.5 + 0.5] + [s + 1.8 + 1.0 for s in [9.44, 19.04, 29.1, 39.09, 49.05]]


def iend(n):
    m = re.search(r"\|i(\d+)\|", n)
    if m: return ends[int(m.group(1)) - 1]
    return stepend[int(re.search(r"step(\d)", n).group(1))]


tend = np.array([iend(n) for n in In])
is_tr = np.array(["step" in n for n in In])


def folds(g, seed, k=4):
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(g)); size = np.zeros(k); asg = {}
    for gi in u:
        j = int(np.argmin(size)); asg[gi] = j; size[j] += (g == gi).sum()
    fid = np.array([asg[gi] for gi in g])
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(k)]


def rf(): return RandomForestClassifier(300, n_jobs=-1, random_state=0)


def cvpred(X, yt, g, seeds=(0, 1, 2)):
    P = []
    for s in seeds:
        p = np.empty_like(yt)
        for tr, te in folds(g, s):
            p[te] = rf().fit(X[tr], yt[tr]).predict(X[te])
        P.append(p)
    return P


def acc(P, yt, mask=None):
    m = np.ones(len(yt), bool) if mask is None else mask
    return np.mean([accuracy_score(yt[m], p[m]) for p in P])


print("== A. ablation (accumulator, RF, block-CV) ==")
for name, X in [("whole", W), ("interval stats only", I[:, ~is_tr]), ("step-transient only", I[:, is_tr]), ("all interval", I)]:
    print(f"{name:22s} {acc(cvpred(X, ya, grp), ya):.3f}  (n_feat {X.shape[1]})")

print("== B. stricter grouping: runs of accumulator label only ==")
ga = np.cumsum(np.r_[True, ya[1:] != ya[:-1]]) - 1
print("accumulator-runs:", ga.max() + 1)
for name, X in [("whole", W), ("interval", I)]:
    print(f"{name:10s} {acc(cvpred(X, ya, ga), ya):.3f}")

print("== C. leave-one-cooler-grade-out (unseen oil temperature) ==")
for name, X in [("whole", W), ("interval", I)]:
    row = []
    for c in [3, 20, 100]:
        tr, te = cool != c, cool == c
        row.append(accuracy_score(ya[te], rf().fit(X[tr], ya[tr]).predict(X[te])))
    print(f"{name:10s} cooler3 {row[0]:.3f}  cooler20 {row[1]:.3f}  cooler100 {row[2]:.3f}")

print("== D. block-CV accuracy split by cooler grade and stability ==")
for name, X in [("whole", W), ("interval", I)]:
    P = cvpred(X, ya, grp)
    s = " ".join(f"c{c}:{acc(P, ya, cool == c):.3f}" for c in [3, 20, 100])
    print(f"{name:10s} {s}  stable:{acc(P, ya, stable == 0):.3f} unstable:{acc(P, ya, stable == 1):.3f}")

print("== E. early decision: features available by time t (accumulator / valve) ==")
for t in [2, 12, 22, 32, 42, 52, 61]:
    m = tend <= t
    if m.sum() == 0: continue
    print(f"t<={t:2d}s  n_feat {m.sum():4d}  acc {acc(cvpred(I[:, m], ya, grp, (0,)), ya):.3f}  valve {acc(cvpred(I[:, m], y[:, 1], grp, (0,)), y[:, 1]):.3f}")

print("== F. what carries accumulator info (perm. importance grouped) ==")
tr, te = folds(grp, 0)[0]
m = rf().fit(I[tr], ya[tr])
pi = permutation_importance(m, I[te], ya[te], n_repeats=5, random_state=0, n_jobs=-1).importances_mean
from collections import defaultdict
bys, byi = defaultdict(float), defaultdict(float)
for n, v in zip(In, pi):
    parts = n.split("|"); bys[parts[0]] += v; byi[parts[1]] += v
print("sensor:", sorted(((k, round(v, 3)) for k, v in bys.items()), key=lambda x: -x[1])[:6])
print("interval:", sorted(((k, round(v, 3)) for k, v in byi.items()), key=lambda x: -x[1])[:8])
top = np.argsort(-pi)[:10]
print("top feats:", [(In[i], round(pi[i], 3)) for i in top])
