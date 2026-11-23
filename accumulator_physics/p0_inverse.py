"""Physics-based inverse estimate of accumulator precharge p0 from the PS1 charging ramp.

Rig facts (Helwig 2018 diss.): V10 closed during the first ~1 s -> pump flow goes into the accumulator
(FS1 = 0); accumulator V0 = 0.6 L (Hydac SBO330); pump 7.5 L/min; PS1 sits next to the accumulator.
Gas polytropic: p * Vg^n = p0 * V0^n  ->  stored oil Voil(p) = V0 * (1 - (p0/p)^(1/n))
With constant charging flow Q:  V0 * p0^(1/n) * [p_s^(-1/n) - p(t)^(-1/n)] = Q * (t - t_s)
=> X(t) = p_s^(-1/n) - p^(-1/n) is linear in t with slope s = Q / (V0 * p0^(1/n))
=> p0 = (Q / (V0 * s))^n          (pressures absolute, bar)
Gas temperature: p0 at oil temperature T relates to fill pressure at 20 C by p0_20 = p0_T * 293.15 / (T + 273.15).
"""
import numpy as np

d = np.load("hyd.npz")
y = d["profile"]
P = d["PS1"].astype(float) + 1.0  # absolute bar
FS1 = d["FS1"].astype(float)
T = d["TS1"].astype(float)[:, :2].mean(1)  # oil temperature at cycle start
V0 = 0.6
t = np.arange(150) / 100.0
Qnom = 7.5 / 60  # L/s
Qeff = np.median(FS1[:, 309:389], 1) / 60  # flow at the ~167 bar steady load step (pump flow minus leakage)
lo, hi = 156.0, 186.0

out = {}
for n in (1.0, 1.4):
    s_all = np.full(len(P), np.nan); r2 = np.full(len(P), np.nan)
    for i in range(len(P)):
        p = P[i, :150]
        k = np.flatnonzero((p >= lo) & (p <= hi))
        if len(k) < 8: continue
        k = k[(k >= k[0]) & (k <= k[0] + np.argmax(p[k[0]:] >= hi) if np.any(p[k[0]:] >= hi) else k[-1])]
        X = p[k[0]] ** (-1 / n) - p[k] ** (-1 / n)
        A = np.c_[t[k] - t[k[0]], np.ones(len(k))]
        coef, res, *_ = np.linalg.lstsq(A, X, rcond=None)
        s_all[i] = coef[0]
        r2[i] = 1 - ((X - A @ coef) ** 2).sum() / max(((X - X.mean()) ** 2).sum(), 1e-30)
    for qn, Q in (("Qnom", Qnom), ("Qeff", Qeff)):
        p0 = (Q / (V0 * s_all)) ** n - 1.0  # back to gauge bar
        out[(n, qn, "T")] = p0
        out[(n, qn, "20C")] = (p0 + 1) * 293.15 / (T + 273.15) - 1
    print(f"n={n}: ramp fit R^2 median {np.nanmedian(r2):.4f}, 5th pct {np.nanpercentile(r2, 5):.4f}, failed fits {np.isnan(s_all).sum()}")

classes = np.array([90, 100, 115, 130])
ya, cool = y[:, 3], y[:, 0]
print("\nMedian estimated p0 [bar] per true class, by cooler grade (oil temp)")
for key in [(1.4, "Qnom", "T"), (1.4, "Qeff", "T"), (1.4, "Qeff", "20C"), (1.0, "Qeff", "20C")]:
    p0 = out[key]
    print(f"  n={key[0]} {key[1]} {key[2]:3s}: " + " | ".join(
        f"c{c}: " + " ".join(f"{np.nanmedian(p0[(cool == c) & (ya == a)]):6.1f}" for a in classes) for c in [3, 20, 100]))

# physics-only classification: no training, nearest nominal class
print("\nPhysics-only accuracy (nearest nominal p0, no training data used)")
for key, p0 in out.items():
    pred = classes[np.argmin(np.abs(np.nan_to_num(p0, nan=0)[:, None] - classes[None]), 1)]
    accs = [np.mean(pred[cool == c] == ya[cool == c]) for c in [3, 20, 100]]
    print(f"  n={key[0]} {key[1]} {key[2]:3s}: c3 {accs[0]:.3f} c20 {accs[1]:.3f} c100 {accs[2]:.3f} all {np.mean(pred == ya):.3f}")
np.savez_compressed("p0.npz", **{f"p0_n{k[0]}_{k[1]}_{k[2]}": v for k, v in out.items()}, Qeff=Qeff, T=T)
