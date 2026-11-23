"""Experiment 9: normal-only calibration of PHYSICAL parameters (not feature offsets).

Charging ramp (V10 closed, FS1 = 0), total compliance = accumulator + line/hose compliance C:
    Q * dt = [ dVacc/dp + C ] dp,    Vacc(p) = V0 (1 - (p0/p)^(1/n))
=> Q * (t - t_s) = a * X(p) + C * (p - p_s),   a = V0 * p0^(1/n),  X = p_s^(-1/n) - p^(-1/n)
=> (t - t_s) = (a/Q) X + (C/Q) (p - p_s)       (linear in X and dp)
Rig parameters Q (effective charging flow) and C (line compliance) are calibrated ONLY on healthy cycles
(accumulator 130 bar known, pump no leak, valve 100 %). Then p0 of every cycle is solved with Q, C fixed.
No fault labels are used anywhere. Gas temperature: p0 known at 20 C fill -> p0_T = p0_20 * (T+273.15)/293.15.
"""
import numpy as np

d = np.load("hyd.npz")
y = d["profile"]
P = d["PS1"].astype(float) + 1.0  # absolute bar
Tc = d["TS1"].astype(float)[:, :2].mean(1)
V0, n = 0.6, 1.4
lo, hi = 153.0, 187.0
ya, cool, pump, valve = y[:, 3], y[:, 0], y[:, 2], y[:, 1]
classes = np.array([90, 100, 115, 130])
healthy = (ya == 130) & (pump == 0) & (valve == 100)


def ramp(i):
    p = P[i, :150]
    k0 = np.argmax(p >= lo)
    if p[k0] < lo: return None
    k1 = k0 + np.argmax(p[k0:] >= hi)
    if k1 - k0 < 8: return None
    k = np.arange(k0, k1 + 1)
    t = (k - k0) / 100.0
    X = p[k0] ** (-1 / n) - p[k] ** (-1 / n)
    dp = p[k] - p[k0]
    return t, X, dp


R = [ramp(i) for i in range(len(P))]
ok = np.array([r is not None for r in R])


def calibrate(idx):
    """joint least squares over healthy cycles: t = (a_i/Q) X + (C/Q) dp, a_i known."""
    rows_A, rows_b = [], []
    for i in idx:
        if R[i] is None: continue
        t, X, dp = R[i]
        p0abs = (130 + 1) * (Tc[i] + 273.15) / 293.15
        a = V0 * p0abs ** (1 / n)
        # unknowns u1 = 1/Q, u2 = C/Q :  t = u1 * a * X + u2 * dp
        rows_A.append(np.c_[a * X, dp]); rows_b.append(t)
    A = np.vstack(rows_A); b = np.concatenate(rows_b)
    (u1, u2), *_ = np.linalg.lstsq(A, b, rcond=None)
    return 1 / u1, u2 / u1  # Q [L/s], C [L/bar]


def estimate(Q, C):
    p0 = np.full(len(P), np.nan)
    for i in range(len(P)):
        if R[i] is None: continue
        t, X, dp = R[i]
        rhs = Q * t - C * dp          # = a * X
        a = (X @ rhs) / (X @ X)
        if a <= 0: continue
        p0abs_T = (a / V0) ** n
        p0[i] = p0abs_T * 293.15 / (Tc[i] + 273.15) - 1  # 20 C fill, gauge
    return p0


def score(p0, mask):
    v = np.nan_to_num(p0, nan=0.0)
    pred = classes[np.argmin(np.abs(v[:, None] - classes[None]), 1)]
    return np.mean(pred[mask] == ya[mask]), np.nanmean(np.abs(p0[mask] - ya[mask]))


print("healthy cycles per cooler:", {c: int((healthy & (cool == c)).sum()) for c in [3, 20, 100]})

# baseline: nominal Q, no line compliance (previous experiment 7)
p_nom = estimate(7.5 / 60, 0.0)
print("\n[baseline] nominal Q=7.5 L/min, C=0 (no calibration):")
for c in [3, 20, 100]:
    a, e = score(p_nom, cool == c); print(f"   cooler {c}: acc {a:.3f}  MAE {e:5.1f} bar")

print("\n[S1] calibrate Q, C on healthy cycles of the OTHER two temperatures (zero target data):")
for c in [3, 20, 100]:
    Q, C = calibrate(np.flatnonzero(healthy & (cool != c)))
    p0 = estimate(Q, C); a, e = score(p0, cool == c)
    print(f"   target cooler {c}: Q={Q*60:.2f} L/min C={C*1000:.3f} mL/bar -> acc {a:.3f}  MAE {e:5.1f} bar")

print("\n[S2] calibrate on k healthy cycles of the TARGET temperature only (10 random draws):")
for k in [3, 5, 20]:
    accs = {}
    for c in [3, 20, 100]:
        pool = np.flatnonzero(healthy & (cool == c)); r_acc = []
        for r in range(10):
            rng = np.random.default_rng(r)
            cal = rng.choice(pool, min(k, len(pool)), replace=False)
            p0 = estimate(*calibrate(cal)); m = (cool == c); m[cal] = False
            r_acc.append(score(p0, m)[0])
        accs[c] = np.mean(r_acc)
    print(f"   k={k:2d}: c3 {accs[3]:.3f} c20 {accs[20]:.3f} c100 {accs[100]:.3f}  mean {np.mean(list(accs.values())):.3f}")

print("\n[S3] per-temperature calibration with all its healthy cycles: estimated p0 medians (true 90/100/115/130)")
allp = np.full(len(P), np.nan)
for c in [3, 20, 100]:
    Q, C = calibrate(np.flatnonzero(healthy & (cool == c)))
    p0 = estimate(Q, C); allp[cool == c] = p0[cool == c]
    med = [np.nanmedian(p0[(cool == c) & (ya == a) & (pump == 0)]) for a in classes]
    medL = [np.nanmedian(p0[(cool == c) & (ya == a) & (pump == 2)]) for a in classes]
    a_, e_ = score(p0, cool == c)
    print(f"   cooler {c}: Q={Q*60:.2f} C={C*1000:.3f} mL/bar | pump0: {np.round(med,1)} | pump2: {np.round(medL,1)} | acc {a_:.3f} MAE {e_:.1f}")
np.save("p0_physcal.npy", allp)
