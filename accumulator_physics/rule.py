"""Single-signal threshold rule vs ML, under leave-one-cooler-grade-out (and in-condition block-CV).
Rule = best single feature + thresholds (decision tree on 1 feature, leaves = #classes),
feature chosen on TRAINING data only."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

f = np.load("feats.npz")
W, Wn, I, In, y = f["W"], f["Wn"], f["I"], f["In"], f["y"]
X = np.hstack([W, I]); Xn = np.r_[Wn, In]
cool = y[:, 0]
labels = y[:, :4]
grp = np.cumsum(np.r_[True, np.any(labels[1:] != labels[:-1], 1)]) - 1
rng = np.random.default_rng(0); u = rng.permutation(np.unique(grp)); size = np.zeros(4); asg = {}
for gi in u:
    j = int(np.argmin(size)); asg[gi] = j; size[j] += (grp == gi).sum()
fid = np.array([asg[gi] for gi in grp])


def rule_fit(Xtr, ytr):
    k = len(np.unique(ytr)); best = (-1, None, None)
    for j in range(Xtr.shape[1]):
        t = DecisionTreeClassifier(max_leaf_nodes=k, random_state=0).fit(Xtr[:, [j]], ytr)
        a = t.score(Xtr[:, [j]], ytr)
        if a > best[0]: best = (a, j, t)
    return best


def evaluate(split, yt, name):
    r_acc, m_acc, feats = [], [], []
    for tr, te in split:
        a, j, t = rule_fit(X[tr], yt[tr])
        r_acc.append(accuracy_score(yt[te], t.predict(X[te][:, [j]]))); feats.append(Xn[j])
        m = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(X[tr], yt[tr])
        m_acc.append(accuracy_score(yt[te], m.predict(X[te])))
    return np.round(r_acc, 3), np.round(m_acc, 3), feats


coolout = [(np.flatnonzero(cool != c), np.flatnonzero(cool == c)) for c in [3, 20, 100]]
block = [(np.flatnonzero(fid != j), np.flatnonzero(fid == j)) for j in range(4)]
for nm, ti in [("cooler", 0), ("pump", 2), ("valve", 1), ("accumulator", 3)]:
    yt = y[:, ti]
    r, m, fs = evaluate(block, yt, nm)
    print(f"{nm:12s} [block-CV]   rule {r.mean():.3f}  ML {m.mean():.3f}   rule feature: {sorted(set(fs))[:3]}")
    if ti != 0:
        r, m, fs = evaluate(coolout, yt, nm)
        print(f"{'':12s} [new temp]   rule {r} mean {r.mean():.3f} | ML {m} mean {m.mean():.3f}   rule feature: {fs}")
