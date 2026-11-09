"""Generate reference cutoffs with cssa.py for check_port.js (JS vs Python equality test)."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cssa import RobustCalibrator, radius_bernstein, box_bounds
rng = np.random.default_rng(7); cases = []
for t in range(200):
    n = int(rng.integers(5, 300)); G = float(rng.choice([1.0, 2, 3, 5, 8])); p = rng.uniform(0.1, 0.9, n)
    s = rng.gamma(2, 1, n)
    if t % 5 == 0: s = np.round(s, 1)
    eps = float(rng.choice([0.0, 0.05, 0.2, -1]))
    epsv = np.inf if eps < 0 else (radius_bernstein(p, G, 0.05) if t % 3 == 0 else eps)
    a = float(rng.choice([0.05, 0.1, 0.2])); h = float(box_bounds(rng.uniform(.1, .9), G)[1])
    q = RobustCalibrator(s, p, G, epsv).cutoff(a, h)
    cases.append(dict(s=s.tolist(), p=p.tolist(), G=G, eps=None if not np.isfinite(epsv) else epsv, a=a, h=h,
                      q=None if not np.isfinite(q) else float(q), rb=float(radius_bernstein(p, G, 0.05))))
json.dump(cases, open(os.path.join(HERE, "port_cases.json"), "w")); print("wrote", len(cases), "cases")
