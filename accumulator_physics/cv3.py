import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

f = np.load("feats.npz")
W, I, y = f["W"], f["I"], f["y"]
ya = y[:, 3]
ga = np.cumsum(np.r_[True, ya[1:] != ya[:-1]]) - 1
print("accumulator runs (start cycle, length, class):")
for g in np.unique(ga):
    idx = np.flatnonzero(ga == g)
    print(f"  run{g:2d} start {idx[0]:4d} len {len(idx):4d} class {ya[idx[0]]:.0f}  cooler {sorted(set(y[idx,0].astype(int)))}")
# order of the other labels to understand nesting
for c, nm in [(0, "cooler"), (1, "valve"), (2, "pump")]:
    r = np.cumsum(np.r_[True, y[1:, c] != y[:-1, c]]) - 1
    print(nm, "runs:", r.max() + 1)

print("== leave-one-accumulator-run-out (other runs of same class stay in training) ==")
for name, X in [("whole", W), ("interval", I)]:
    accs, ns = [], []
    allp, ally = [], []
    for g in np.unique(ga):
        te = ga == g
        p = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(X[~te], ya[~te]).predict(X[te])
        accs.append(accuracy_score(ya[te], p)); ns.append(te.sum()); allp.append(p); ally.append(ya[te])
    print(name, "per-run acc:", [round(a, 2) for a in accs], " weighted", round(np.average(accs, weights=ns), 3))
    print(confusion_matrix(np.concatenate(ally), np.concatenate(allp), labels=[90, 100, 115, 130]))

print("== does cycle index alone predict accumulator? (time confound) ==")
labels = y[:, :4]
grp = np.cumsum(np.r_[True, np.any(labels[1:] != labels[:-1], 1)]) - 1
rng = np.random.default_rng(0); u = rng.permutation(np.unique(grp)); size = np.zeros(4); asg = {}
for gi in u:
    j = int(np.argmin(size)); asg[gi] = j; size[j] += (grp == gi).sum()
fid = np.array([asg[gi] for gi in grp])
t = np.arange(len(ya))[:, None].astype(float)
for name, X in [("cycle index only", t)]:
    p = np.empty_like(ya)
    for j in range(4):
        tr, te = fid != j, fid == j
        p[te] = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(X[tr], ya[tr]).predict(X[te])
    print(name, "block-CV acc", round(accuracy_score(ya, p), 3))
