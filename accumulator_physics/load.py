import numpy as np, sys
raw=sys.argv[1]; out=sys.argv[2]
names="PS1 PS2 PS3 PS4 PS5 PS6 EPS1 FS1 FS2 TS1 TS2 TS3 TS4 VS1 CE CP SE".split()
d={n:np.loadtxt(f"{raw}/{n}.txt",dtype=np.float32) for n in names}
d["profile"]=np.loadtxt(f"{raw}/profile.txt")
np.savez_compressed(out,**d)
p=d["PS1"].mean(0); dp=np.abs(np.diff(p))
# print PS1 mean every 0.5 s
for t in range(0,6000,50): print(t/100, round(float(p[t]),1), round(float(d["PS2"].mean(0)[t]),1), round(float(d["FS1"].mean(0)[t//10]),2))
