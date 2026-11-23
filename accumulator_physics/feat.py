import numpy as np, scipy.stats as st
d = np.load("hyd.npz")
names = "PS1 PS2 PS3 PS4 PS5 PS6 EPS1 FS1 FS2 TS1 TS2 TS3 TS4 VS1 CE CP SE".split()

# load-step times from mean PS1 derivative
p = d["PS1"].mean(0)
g = np.abs(np.gradient(np.convolve(p, np.ones(5) / 5, "same")))
steps = [lo + int(np.argmax(g[lo:hi])) for lo, hi in [(800, 1100), (1800, 2050), (2800, 3050), (3800, 4050), (4800, 5050)]]
print("steps (s):", [s / 100 for s in steps])


def stats11(x):  # Enejor-style whole-cycle descriptors
    k = max(1, x.shape[1] // 10)
    f = [x.mean(1), x.std(1), x.min(1), x.max(1), np.median(x, 1), np.ptp(x, 1),
         np.nan_to_num(st.skew(x, 1)), np.nan_to_num(st.kurtosis(x, 1)),
         x[:, :k].mean(1), x[:, -k:].mean(1), x[:, -k:].mean(1) - x[:, :k].mean(1)]
    return np.stack(f, 1)


W, Wn = [], []
for n in names:
    W.append(stats11(d[n].astype(float)))
    Wn += [f"{n}|{s}" for s in "mean std min max med rng skew kurt first10 last10 trend".split()]
W = np.hstack(W)

edges = [0, 1.5]
for s in steps:
    edges += [s / 100 - 0.2, s / 100 + 1.8]
edges.append(60.0)
iv = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
print(len(iv), "intervals:", [(round(a, 1), round(b, 1)) for a, b in iv])
rate = {n: (100 if n[:2] in ("PS", "EP") else 10 if n[:2] == "FS" else 1) for n in names}


def ivstats(x):
    tc = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    slope = (x * tc).sum(1) / (tc ** 2).sum()
    return np.stack([np.median(x, 1), x.var(1), slope, np.nan_to_num(st.skew(x, 1)),
                     np.nan_to_num(st.kurtosis(x, 1)), x.argmax(1) / x.shape[1]], 1)


I, In = [], []
for n in names:
    x = d[n].astype(float); r = rate[n]
    for j, (a, b) in enumerate(iv):
        seg = x[:, int(a * r):int(b * r)]
        if seg.shape[1] >= 8:
            I.append(ivstats(seg)); In += [f"{n}|i{j+1}|{s}" for s in "med var slope skew kurt pmax".split()]
        elif seg.shape[1] >= 1:
            I.append(seg.mean(1)[:, None]); In.append(f"{n}|i{j+1}|mean")

# transient (step-response) features for fast sensors
T, Tn = [], []
for n in ["PS1", "PS2", "PS3", "EPS1", "FS1"]:
    x = d[n].astype(float); r = rate[n]
    for k, s in enumerate([0] + steps):
        a = int(s / 100 * r); w = int(2.0 * r)
        seg = x[:, a:a + w]
        pre = x[:, max(a - int(0.3 * r), 0):a + 1].mean(1) if a > 0 else seg[:, 0]
        fin = x[:, a + w:a + w + r].mean(1)
        dlt = fin - pre; dev = seg - fin[:, None]
        den = np.where(np.abs(dlt) < 1e-6, 1e-6, dlt)[:, None]
        frac = (seg - pre[:, None]) / den
        t90 = np.argmax(frac >= 0.9, 1) / r; t10 = np.argmax(frac >= 0.1, 1) / r
        over = np.max(dev * np.sign(dlt)[:, None], 1)
        iae = np.abs(dev).sum(1) / r
        df = np.diff(seg, axis=1) * r
        T.append(np.stack([dlt, t10, t90, t90 - t10, over, iae, df.max(1), df.min(1)], 1))
        Tn += [f"{n}|step{k}|{s}" for s in "delta t10 t90 rise over iae dmax dmin".split()]
I = np.hstack(I + T); In += Tn
np.savez_compressed("feats.npz", W=W, Wn=np.array(Wn), I=I, In=np.array(In), y=d["profile"])
print("W", W.shape, "I", I.shape)
