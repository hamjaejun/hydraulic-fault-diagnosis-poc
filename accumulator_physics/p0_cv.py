import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

f = np.load("feats.npz"); g = np.load("phys.npz"); h = np.load("p0.npz")
I, In, y = f["I"], f["In"], f["y"]
Kn = g["Kn"]; Kps1 = g["K"][:, [i for i, n in enumerate(Kn) if n.startswith("K|PS1")]]
T = h["T"]
p0T = np.nan_to_num(h["p0_n1.4_Qnom_T"], nan=np.nanmedian(h["p0_n1.4_Qnom_T"]))
p020 = np.nan_to_num(h["p0_n1.4_Qnom_20C"], nan=np.nanmedian(h["p0_n1.4_Qnom_20C"]))
ya, cool = y[:, 3], y[:, 0]
classes = np.array([90, 100, 115, 130])
timing = np.array([n.split("|")[-1] in ("t10", "t90", "rise") for n in In])
Tm = I[:, timing]


def nearest(v):
    return classes[np.argmin(np.abs(v[:, None] - classes[None]), 1)]


print("== A. physics estimate + linear calibration learned on the other two temperatures ==")
for nm, X in [("p0_hat(T)", p0T[:, None]), ("p0_hat(20C gas-temp corr.)", p020[:, None]),
              ("p0_hat(T) + oil temp", np.c_[p0T, T])]:
    accs = []
    for c in [3, 20, 100]:
        tr, te = cool != c, cool == c
        m = LinearRegression().fit(X[tr], ya[tr])
        accs.append(accuracy_score(ya[te], nearest(m.predict(X[te]))))
    print(f"  {nm:28s} c3 {accs[0]:.3f} c20 {accs[1]:.3f} c100 {accs[2]:.3f} mean {np.mean(accs):.3f}")


def coolout(X, seeds=(0, 1, 2)):
    r = []
    for c in [3, 20, 100]:
        tr, te = cool != c, cool == c
        r.append(np.mean([accuracy_score(ya[te], RandomForestClassifier(300, n_jobs=-1, random_state=s).fit(X[tr], ya[tr]).predict(X[te])) for s in seeds]))
    return np.array(r)


print("== B. RF, leave-one-cooler-out (3 seeds) ==")
sets = {"p0_hat only (1)": p0T[:, None],
        "p0_hat + T (2)": np.c_[p0T, T],
        "K gas-law (6)  [prev]": Kps1,
        "timing + K  [prev best]": np.c_[Tm, Kps1],
        "p0_hat + K + T": np.c_[p0T, p020, T, Kps1],
        "p0_hat + timing + K + T": np.c_[p0T, p020, T, Tm, Kps1]}
for nm, X in sets.items():
    r = coolout(X)
    print(f"  {nm:26s} c3 {r[0]:.3f} c20 {r[1]:.3f} c100 {r[2]:.3f} mean {r.mean():.3f}")

print("== C. continuous-error view: MAE of calibrated p0 (bar), linear calib on other temps ==")
for c in [3, 20, 100]:
    tr, te = cool != c, cool == c
    m = LinearRegression().fit(np.c_[p0T, T][tr], ya[tr]); e = np.abs(m.predict(np.c_[p0T, T][te]) - ya[te])
    print(f"  cooler {c}: MAE {e.mean():.1f} bar, within 7.5 bar {np.mean(e < 7.5):.3f}")
