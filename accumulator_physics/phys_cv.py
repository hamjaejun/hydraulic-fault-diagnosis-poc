import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

f = np.load("feats.npz"); g = np.load("phys.npz")
W, I, In, y = f["W"], f["I"], f["In"], f["y"]
K, Kn, V = g["K"], g["Kn"], g["V"]
cool = y[:, 0]
timing = np.array([n.split("|")[-1] in ("t10", "t90", "rise") for n in In])
T = I[:, timing]
Kps1 = K[:, [i for i, n in enumerate(Kn) if n.startswith("K|PS1")]]


def coolout(X, yt, seeds=(0, 1, 2)):
    out = []
    for c in [3, 20, 100]:
        tr, te = cool != c, cool == c
        out.append(np.mean([accuracy_score(yt[te], RandomForestClassifier(300, n_jobs=-1, random_state=s).fit(X[tr], yt[tr]).predict(X[te])) for s in seeds]))
    return np.array(out)


sets = {
    "whole (Enejor-type)": W,
    "timing-only": T,
    "K gas-law only (PS1, 6)": Kps1,
    "K gas-law all (36)": K,
    "timing + K": np.hstack([T, K]),
    "timing + K + viscosity": np.hstack([T, K, V]),
    "all interval": I,
    "all interval + K + visc": np.hstack([I, K, V]),
}
for tn, ti in [("accumulator", 3), ("pump", 2)]:
    print(f"== {tn}: leave-one-cooler-out (c3 hot / c20 / c100 cold), RF x3 seeds ==")
    sel = sets if tn == "accumulator" else {k: sets[k] for k in ["whole (Enejor-type)", "all interval"]} | {"whole + viscosity": np.hstack([W, V]), "all interval + K + visc": sets["all interval + K + visc"]}
    for nm, X in sel.items():
        r = coolout(X, y[:, ti])
        print(f"  {nm:26s} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f}  mean {r.mean():.3f}  (n={X.shape[1]})", flush=True)
