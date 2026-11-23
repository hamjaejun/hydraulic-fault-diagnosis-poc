"""PHM 2022 rock drill: cross-individual transfer, accumulator classes (5 = return accumulator damaged,
11 = high-pressure accumulator charge low). Leave-one-individual-out (5 training individuals).
Scenarios: zero-shot, normal-reference subtraction (per-individual ref cycles, no fault labels),
few-shot (k labelled cycles per class from the target individual)."""
import zipfile, numpy as np, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

zpath = sys.argv[1]
z = zipfile.ZipFile(zpath); base = "Data_Challenge_PHM2022_training_data/"
L = 556  # shortest cycle length in the dataset description
inds = ["1", "2", "4", "5", "6"]


def load(prefix, ind):
    rows = z.read(f"{base}{prefix}_{ind}.csv".replace("_" + ind, ind) if False else f"{base}{prefix}{ind}.csv").decode().strip().split("\n")
    lab = np.array([int(float(r.split(",", 1)[0])) for r in rows])
    X = np.array([np.array(r.split(",")[1:L + 1], float) for r in rows])
    return lab, X


X, y, g, R = [], [], [], {}
for ind in inds:
    parts = []; ref = []
    for s in ["pin", "pdmp", "po"]:
        lab, A = load(f"data_{s}", ind); parts.append(A)
        _, B = load(f"ref_{s}", ind); ref.append(B)
    Xi = np.hstack(parts)
    X.append(Xi); y.append(lab); g.append(np.full(len(lab), int(ind)))
    R[int(ind)] = np.hstack(ref).mean(0)
X = np.vstack(X); y = np.concatenate(y); g = np.concatenate(g)
print("cycles", X.shape, "ref cycles per individual used as normal reference")
Xr = X - np.vstack([R[i] for i in g])  # subtract individual's mean normal waveform


def rf(s=0): return RandomForestClassifier(200, n_jobs=-1, random_state=s, max_features="sqrt")


def evaluate(Xuse, k=0, reps=1):
    acc, r5, r11 = [], [], []
    for t in sorted(set(g)):
        for r in range(reps):
            tr = g != t; te = g == t
            if k > 0:
                rng = np.random.default_rng(r); add = []
                for c in np.unique(y[te]):
                    add.extend(rng.choice(np.flatnonzero(te & (y == c)), k, replace=False))
                add = np.array(add); tr = tr.copy(); tr[add] = True; te = te.copy(); te[add] = False
            p = rf(r).fit(Xuse[tr], y[tr]).predict(Xuse[te])
            acc.append(accuracy_score(y[te], p))
            r5.append(recall_score(y[te], p, labels=[5], average="macro"))
            r11.append(recall_score(y[te], p, labels=[11], average="macro"))
    return np.mean(acc), np.mean(r5), np.mean(r11), np.array(acc)


print("\n11-class accuracy, leave-one-individual-out (mean over 5 targets); recall of accumulator classes")
for name, Xuse, k, reps in [("zero-shot (raw)", X, 0, 1),
                            ("normal-ref subtraction (no fault labels)", Xr, 0, 1),
                            ("few-shot k=2/class (raw)", X, 2, 2),
                            ("few-shot k=10/class (raw)", X, 10, 2),
                            ("normal-ref + few-shot k=2/class", Xr, 2, 2)]:
    a, r5, r11, per = evaluate(Xuse, k, reps)
    print(f"  {name:42s} acc {a:.3f}  recall[ret.acc dmg] {r5:.3f}  recall[HP acc low charge] {r11:.3f}  per-target {np.round(per[::reps],3)}", flush=True)
