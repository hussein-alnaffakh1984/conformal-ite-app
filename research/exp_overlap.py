"""Where does the moment constraint help? The cutoffs depend on the data only through the
calibration propensities and the ranks of the scores, so the gain can be studied without an
outcome model. Scores are |N(0,1)| independent of e (worst-case-free baseline); e ~ Uniform(a,b).
Gain = 1 - mean(MC cutoff)/mean(box cutoff), matched guarantee 1 - alpha (MC at alpha - delta)."""
import json, numpy as np, os
from cssa import RobustCalibrator, box_bounds, radius_bernstein
ROOT = os.path.dirname(os.path.abspath(__file__))
DELTA, N, REPS = 0.001, 1366, 200
def run(lo, hi, G, alpha, seed0=0):
    gb, gm = [], []
    for r in range(REPS):
        rng = np.random.default_rng(seed0 + r)
        p = rng.uniform(lo, hi, N); s = np.abs(rng.standard_normal(N))
        pt = rng.uniform(lo, hi, 200); h = box_bounds(pt, G)[1]
        qb = RobustCalibrator(s, p, G, np.inf).cutoffs_for_tests(alpha, h)
        qm = RobustCalibrator(s, p, G, radius_bernstein(p, G, DELTA)).cutoffs_for_tests(alpha - DELTA, h)
        gb.append(np.mean(qb)); gm.append(np.mean(qm))
    g = 100 * (1 - np.array(gm) / np.array(gb))
    return float(g.mean()), float(g.std(ddof=1) / np.sqrt(REPS))
out = []
for lo, hi in [(0.05, 0.30), (0.19, 0.77), (0.40, 0.60), (0.70, 0.95)]:
    for G in (3.0, 5.0):
        out.append(dict(range=[lo, hi], G=G, alpha=0.10, gain=run(lo, hi, G, 0.10)))
for a in (0.05, 0.20):
    out.append(dict(range=[0.19, 0.77], G=5.0, alpha=a, gain=run(0.19, 0.77, 5.0, a)))
json.dump(out, open(os.path.join(ROOT, "results", "overlap_gain.json"), "w"), indent=1)
for o in out: print(o)
