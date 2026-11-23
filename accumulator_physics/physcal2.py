"""Experiment 9b: line compliance came out negative (unphysical) -> calibrate polytropic exponent n and Q instead
(C = 0). Healthy cycles only (accumulator 130, pump 0, valve 100, stable flag 0). No fault labels used."""
import numpy as np

d = np.load("hyd.npz")
y = d["profile"]
P = d["PS1"].astype(float) + 1.0
Tc = d["TS1"].astype(float)[:, :2].mean(1)
V0 = 0.6; lo, hi = 153.0, 187.0
ya, cool, pump, valve, stab = y[:, 3], y[:, 0], y[:, 2], y[:, 1], y[:, 4]
classes = np.array([90, 100, 115, 130])
healthy = (ya == 130) & (pump == 0) & (valve == 100) & (stab == 0)


def ramp(i):
    p = P[i, :150]; k0 = np.argmax(p >= lo)
    if p[k0] < lo: return None
    k1 = k0 + np.argmax(p[k0:] >= hi)
    if k1 - k0 < 8: return None
    k = np.arange(k0, k1 + 1)
    return (k - k0) / 100.0, p[k]


R = [ramp(i) for i in range(len(P))]


def calibrate(idx, ngrid=np.linspace(0.8, 2.0, 61)):
    best = None
    for n in ngrid:
        num = den = 0.0; res = 0.0; rows = []
        for i in idx:
            if R[i] is None: continue
            t, p = R[i]
            a = V0 * ((130 + 1) * (Tc[i] + 273.15) / 293.15) ** (1 / n)
            X = p[0] ** (-1 / n) - p ** (-1 / n)
            rows.append((t, a * X))
        # t = (1/Q) * aX  -> LS for 1/Q
        num = sum((ax @ t) for t, ax in rows); den = sum((ax @ ax) for t, ax in rows)
        u = num / den
        res = sum(((t - u * ax) ** 2).sum() for t, ax in rows)
        if best is None or res < best[0]: best = (res, n, 1 / u)
    return best[1], best[2]


def estimate(n, Q):
    p0 = np.full(len(P), np.nan)
    for i in range(len(P)):
        if R[i] is None: continue
        t, p = R[i]; X = p[0] ** (-1 / n) - p ** (-1 / n)
        a = Q * (X @ t) / (X @ X)
        p0[i] = (a / V0) ** n * 293.15 / (Tc[i] + 273.15) - 1
    return p0


def score(p0, m):
    v = np.nan_to_num(p0, nan=0.0)
    pred = classes[np.argmin(np.abs(v[:, None] - classes[None]), 1)]
    return np.mean(pred[m] == ya[m]), np.nanmean(np.abs(p0[m] - ya[m]))


print("healthy (stable) cycles per cooler:", {c: int((healthy & (cool == c)).sum()) for c in [3, 20, 100]})
print("\n[S1] calibrate (n, Q) on healthy cycles of the OTHER two temperatures:")
for c in [3, 20, 100]:
    n, Q = calibrate(np.flatnonzero(healthy & (cool != c)))
    p0 = estimate(n, Q); a, e = score(p0, cool == c)
    med = [np.nanmedian(p0[(cool == c) & (ya == k) & (pump == 0)]) for k in classes]
    print(f"   target c{c}: n={n:.2f} Q={Q*60:.2f} L/min -> acc {a:.3f} MAE {e:5.1f} | pump0 medians {np.round(med,1)}")

print("\n[S2] calibrate on k healthy cycles of the TARGET temperature (10 draws):")
for k in [3, 5, 20]:
    accs = []
    for c in [3, 20, 100]:
        pool = np.flatnonzero(healthy & (cool == c)); r = []
        for s in range(10):
            cal = np.random.default_rng(s).choice(pool, min(k, len(pool)), replace=False)
            p0 = estimate(*calibrate(cal)); m = cool == c; m[cal] = False
            r.append(score(p0, m)[0])
        accs.append(np.mean(r))
    print(f"   k={k:2d}: c3 {accs[0]:.3f} c20 {accs[1]:.3f} c100 {accs[2]:.3f} mean {np.mean(accs):.3f}")
