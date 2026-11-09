import numpy as np, time
from cssa import RobustCalibrator, tail_mass_lp, box_bounds, radius_hoeffding, radius_bernstein
rng = np.random.default_rng(1)
maxerr = 0.0; cnt = 0; infeas = 0
for trial in range(300):
    n = int(rng.integers(5, 60)); G = float(rng.choice([1.5, 3, 5, 8]))
    p = rng.uniform(0.05, 0.95, n); s = rng.gamma(2, 1, n)
    if trial % 7 == 0: s = np.round(s)          # ties
    eps = float(rng.choice([0.0, 0.05, 0.2, 0.5, np.inf]))
    h_t = float(box_bounds(rng.uniform(0.05, 0.95), G)[1])
    cal = RobustCalibrator(s, p, G, eps)
    for k in rng.integers(1, n + 1, 3):
        a = cal.tail_mass(int(k), h_t); b = tail_mass_lp(s, p, G, eps, int(k), h_t)
        if a == 1.0 and b == 1.0: infeas += 1
        maxerr = max(maxerr, abs(a - b)); cnt += 1
print(f"{cnt} comparisons, max |greedy - LP| = {maxerr:.2e}, infeasible-both = {infeas}")
# monotonicity of M(k) in k and of cutoff in alpha and h_test
s = rng.gamma(2, 1, 400); p = rng.uniform(0.1, 0.9, 400)
cal = RobustCalibrator(s, p, 5.0, 0.2)
Mk = [cal.tail_mass(k, 6.0) for k in range(1, 401)]
print("M(k) non-increasing:", bool(np.all(np.diff(Mk) <= 1e-12)))
qa = [cal.cutoff(a, 6.0) for a in [0.05, 0.1, 0.2]]; qh = [cal.cutoff(0.1, h) for h in [2, 6, 20]]
print("cutoff decreasing in alpha:", qa, " increasing in h_test:", qh)
t0 = time.time(); [cal.cutoff(0.1, 6.0) for _ in range(200)]; print("cutoff time %.2f ms (n=400)" % (1000*(time.time()-t0)/200))
print("radius n=440,G=5 (p~U(.3,.7)): Hoeffding %.3f  Bernstein %.3f" % (
    radius_hoeffding(rng.uniform(.3,.7,440),5,.05), radius_bernstein(rng.uniform(.3,.7,440),5,.05)))
