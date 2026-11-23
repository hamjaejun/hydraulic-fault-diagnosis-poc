"""Physics-guided features without component specs.
(a) Accumulator: isothermal gas law, p_abs*V_gas = p0*V0  ->  oil stored between p_a and p_b:
    dV = p0*V0*(1/p_a - 1/p_b)   =>  K = dV / (1/p_a - 1/p_b) ~ p0*V0  (V0 is a fixed unknown constant).
    dV is approximated by the flow deficit/excess during each load step: integral of (FS1 - FS1_final) dt.
(b) Oil viscosity from temperature (Walther eq., generic ISO VG46 mineral oil), pump leakage ~ dp/mu.
"""
import numpy as np

d = np.load("hyd.npz")
y = d["profile"]
steps = [0.0, 9.44, 19.04, 29.1, 39.09, 49.05]
FS1 = d["FS1"].astype(float)  # 10 Hz, L/min
P = {k: d[k].astype(float) for k in ["PS1", "PS2", "PS3"]}


def win(x, r, a, b):
    return x[:, max(int(a * r), 0):int(b * r)]


feats, names = [], []
for k, s in enumerate(steps):
    q = win(FS1, 10, s - 0.2, s + 3.0)
    qf = win(FS1, 10, s + 3.0, s + 5.0).mean(1)
    dV = (q - qf[:, None]).sum(1) / 10 / 60  # litres
    for pn, x in P.items():
        pa = win(x, 100, s - 0.4, s - 0.05).mean(1) + 1.0 if s > 0 else x[:, 0] + 1.0
        pb = win(x, 100, s + 3.0, s + 5.0).mean(1) + 1.0
        dinv = 1 / np.clip(pa, 1, None) - 1 / np.clip(pb, 1, None)
        K = dV / np.where(np.abs(dinv) < 1e-5, np.sign(dinv + 1e-12) * 1e-5, dinv)
        feats += [K, dV / np.where(np.abs(pb - pa) < 0.5, 0.5, pb - pa)]
        names += [f"K|{pn}|step{k}", f"Clin|{pn}|step{k}"]
Kmat = np.stack(feats, 1)

# (b) viscosity, Walther: loglog(nu+0.7) = A - B*log10(T_K); VG46: 46 cSt@40C, 6.8 cSt@100C
T1, T2, n1, n2 = 313.15, 373.15, 46.0, 6.8
B = (np.log10(np.log10(n1 + .7)) - np.log10(np.log10(n2 + .7))) / (np.log10(T2) - np.log10(T1))
A = np.log10(np.log10(n1 + .7)) + B * np.log10(T1)
nu = lambda Tc: 10 ** (10 ** (A - B * np.log10(Tc + 273.15))) - 0.7
Toil = d["TS1"].astype(float).mean(1)
mu = nu(Toil)
visc, vn = [mu], ["nu_oil"]
# per steady segment: flow, pressure, and leakage-type groups Q*mu, Q*mu/p, SE*mu
segs = [(1.5, 9.2), (11.2, 18.8), (20.8, 28.9), (30.9, 38.9), (40.9, 48.8), (50.8, 60.0)]
for j, (a, b) in enumerate(segs):
    q = win(FS1, 10, a, b).mean(1); p = win(P["PS1"], 100, a, b).mean(1)
    visc += [q * mu, q * mu / np.clip(p, 1, None), (p / np.clip(q, 1e-3, None)) / mu]
    vn += [f"Qmu|s{j}", f"Qmu_p|s{j}", f"R_mu|s{j}"]
se = d["SE"].astype(float).mean(1)
visc += [se * mu]; vn += ["SEmu"]
V = np.stack(visc, 1)
np.savez_compressed("phys.npz", K=Kmat, Kn=np.array(names), V=V, Vn=np.array(vn))

# sanity: median K per accumulator class, separately for each cooler grade (temperature)
print("oil viscosity (cSt) by cooler:", {c: round(float(np.median(mu[y[:, 0] == c])), 1) for c in [3, 20, 100]})
for nm in ["K|PS1|step1", "K|PS1|step2", "K|PS1|step3", "K|PS2|step2", "K|PS3|step2", "Clin|PS1|step2"]:
    j = names.index(nm); v = Kmat[:, j]
    rows = []
    for c in [3, 20, 100]:
        rows.append(" ".join(f"{np.median(v[(y[:,0]==c)&(y[:,3]==a)]):8.2f}" for a in [130, 115, 100, 90]))
    print(f"{nm:15s} acc130..90 | c3: {rows[0]} | c20: {rows[1]} | c100: {rows[2]}")
