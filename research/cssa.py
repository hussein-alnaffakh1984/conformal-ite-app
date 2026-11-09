"""
cssa.py -- Conformal sensitivity analysis under the marginal sensitivity model (MSM)
with a moment constraint on the worst-case weights.

Setting (arm a):  calibration units are drawn from P(. | A=a); the target is the full
population. Under Y(a) _||_ A | (X, U) the likelihood ratio target/calibration of
(X, U, Y(a)) is proportional to w*(X,U) = 1 / e_a(X,U).  With nominal propensity
p(x) = P(A=a | X=x) and the MSM odds-ratio bound Gamma,

    l(x) = 1 + (1 - p)/(p * Gamma)  <=  w*(x,u)  <=  h(x) = 1 + (1 - p) * Gamma / p.

Moment identity (conditional):  E[ w*(X,U) * p(X) | X, A=a ] = 1, so Z_i = w*_i p_i are
independent given the X's, have conditional mean 1 and lie in [p l, p h] subset [1/G, G].

Robust cutoff: tail mass of the (n+1)-augmented weighted distribution, with the test
point placed at +inf with its own weight w_{n+1} in [l_{n+1}, h_{n+1}]:

    M(k) = sup_{w in F}  ( sum_{i: s_i > s_(k)} w_i + w_{n+1} ) / ( sum_i w_i + w_{n+1} ),

    F = { l_i <= w_i <= h_i ,  |(1/n) sum_i w_i p_i - 1| <= eps }   (eps = inf -> box).

q_hat = s_(k*) with k* = min{ k : M(k) <= alpha }  (q_hat = inf if no such k).
The ratio is maximised at w_{n+1} = h_{n+1}; for the calibration weights the maximiser
does not depend on the Dinkelbach multiplier, giving an exact O(n) greedy evaluation
(after one sort).  `tail_mass_lp` solves the Charnes-Cooper LP with HiGHS for checking.
"""
import numpy as np
from scipy.optimize import linprog


def box_bounds(p, gamma):
    p = np.asarray(p, float)
    lo = 1.0 + (1.0 - p) / (p * gamma)
    hi = 1.0 + (1.0 - p) * gamma / p
    return lo, hi


# ---------------------------------------------------------------- radii
def radius_hoeffding(p, gamma, delta):
    """Two-sided Hoeffding, conditional on X: Z_i in [p_i l_i, p_i h_i]."""
    lo, hi = box_bounds(p, gamma)
    rng2 = np.sum((p * hi - p * lo) ** 2)
    n = len(p)
    return np.sqrt(np.log(2.0 / delta) * rng2 / 2.0) / n


def radius_bernstein(p, gamma, delta):
    """Two-sided Bernstein, conditional on X, with the data-free Bhatia-Davis variance
    bound Var(Z_i|X_i) <= (z_i^+ - 1)(1 - z_i^-) (valid because E[Z_i|X_i] = 1).
    P(|sum(Z_i-1)| >= t) <= 2 exp(-t^2 / (2 (V + b t / 3)))."""
    lo, hi = box_bounds(p, gamma)
    m, M = p * lo, p * hi
    V = float(np.sum(np.clip(M - 1.0, 0, None) * np.clip(1.0 - m, 0, None)))   # clip rounding at Gamma = 1
    b = float(np.max(np.clip(np.maximum(M - 1.0, 1.0 - m), 0, None)))
    L = np.log(2.0 / delta)
    # solve t^2 = 2 L (V + b t / 3)  ->  t = (bL/3) + sqrt((bL/3)^2 + 2 L V)
    t = b * L / 3.0 + np.sqrt((b * L / 3.0) ** 2 + 2.0 * L * V)
    return t / len(p)


# ---------------------------------------------------------------- tail mass
class RobustCalibrator:
    """Pre-sorts one calibration set; evaluates M(k) and the robust cutoff fast."""

    def __init__(self, scores, p, gamma, eps):
        o = np.argsort(scores, kind="mergesort")
        self.s = np.asarray(scores, float)[o]
        self.p = np.asarray(p, float)[o]
        self.lo, self.hi = box_bounds(self.p, gamma)
        self.gamma, self.eps, self.n = gamma, eps, len(self.s)
        self.ord_a = np.argsort(-self.p, kind="mergesort")   # by a = p descending
        n = self.n
        if np.isfinite(eps):
            self.Lb, self.Ub = n * (1.0 - eps), n * (1.0 + eps)
        else:
            self.Lb, self.Ub = -np.inf, np.inf

    def _start(self, k):
        """Tail = indices with s > s_(k) (k is 1-based)."""
        return np.searchsorted(self.s, self.s[k - 1], side="right")

    def tail_mass(self, k, h_test):
        start = self._start(k)
        n = self.n
        tail = np.zeros(n, bool); tail[start:] = True
        w = np.where(tail, self.hi, self.lo).astype(float)
        S = float(np.dot(self.p, w))
        if S > self.Ub:                       # must remove a-mass from the tail (largest a first)
            need = S - self.Ub
            idx = self.ord_a[tail[self.ord_a]]
            cap = (self.hi[idx] - self.lo[idx]) * self.p[idx]
            cs = np.cumsum(cap)
            if cs.size == 0 or cs[-1] < need - 1e-12:
                return 1.0                    # F empty at this eps -> conservative
            j = min(int(np.searchsorted(cs, need)), len(cs) - 1)
            w[idx[:j]] = self.lo[idx[:j]]
            rem = need - (cs[j - 1] if j > 0 else 0.0)
            w[idx[j]] -= rem / self.p[idx[j]]
        elif S < self.Lb:                     # must add a-mass to the head (largest a first)
            need = self.Lb - S
            idx = self.ord_a[~tail[self.ord_a]]
            cap = (self.hi[idx] - self.lo[idx]) * self.p[idx]
            cs = np.cumsum(cap)
            if cs.size == 0 or cs[-1] < need - 1e-12:
                return 1.0
            j = min(int(np.searchsorted(cs, need)), len(cs) - 1)
            w[idx[:j]] = self.hi[idx[:j]]
            rem = need - (cs[j - 1] if j > 0 else 0.0)
            w[idx[j]] += rem / self.p[idx[j]]
        num = w[start:].sum() + h_test
        den = w.sum() + h_test
        return num / den

    def cutoff(self, alpha, h_test):
        if alpha <= 0:
            return np.inf
        n = self.n
        if self.tail_mass(n, h_test) > alpha:
            return np.inf
        lo_k, hi_k = 1, n                     # smallest k with M(k) <= alpha
        if self.tail_mass(1, h_test) <= alpha:
            return self.s[0]
        while hi_k - lo_k > 1:
            mid = (lo_k + hi_k) // 2
            if self.tail_mass(mid, h_test) <= alpha:
                hi_k = mid
            else:
                lo_k = mid
        return self.s[hi_k - 1]

    def cutoffs_for_tests(self, alpha, h_tests, n_grid=25):
        """Cutoff is non-decreasing in h_test: evaluate on a grid and round h up (valid)."""
        h_tests = np.asarray(h_tests, float)
        grid = np.unique(np.quantile(h_tests, np.linspace(0, 1, n_grid)))
        grid[-1] = h_tests.max()
        qg = np.array([self.cutoff(alpha, g) for g in grid])
        pos = np.searchsorted(grid, h_tests, side="left")
        pos = np.minimum(pos, len(grid) - 1)
        return qg[pos]


def tail_mass_lp(scores, p, gamma, eps, k, h_test):
    """Reference: Charnes-Cooper LP solved with HiGHS (test weight free in its box).
    variables (y_1..y_n, y_test, t);  max sum_{tail} y + y_test
    s.t. sum y + y_test = 1,  t l <= y <= t h,  t l_t <= y_test <= t h_t,
         n(1-eps) t <= sum p_i y_i <= n(1+eps) t,  t >= 0."""
    o = np.argsort(scores, kind="mergesort")
    s, pp = np.asarray(scores, float)[o], np.asarray(p, float)[o]
    lo, hi = box_bounds(pp, gamma)
    n = len(s)
    start = np.searchsorted(s, s[k - 1], side="right")
    l_t = 1.0  # any lower bound >=1 for the test weight; optimum uses h_test anyway
    nv = n + 2
    c = np.zeros(nv); c[start:n] = -1.0; c[n] = -1.0
    A_eq = np.zeros((1, nv)); A_eq[0, :n + 1] = 1.0
    rows, b = [], []
    for i in range(n):
        r = np.zeros(nv); r[i] = 1.0; r[-1] = -hi[i]; rows.append(r); b.append(0.0)
        r = np.zeros(nv); r[i] = -1.0; r[-1] = lo[i]; rows.append(r); b.append(0.0)
    r = np.zeros(nv); r[n] = 1.0; r[-1] = -h_test; rows.append(r); b.append(0.0)
    r = np.zeros(nv); r[n] = -1.0; r[-1] = l_t; rows.append(r); b.append(0.0)
    if np.isfinite(eps):
        r = np.zeros(nv); r[:n] = pp; r[-1] = -n * (1 + eps); rows.append(r); b.append(0.0)
        r = np.zeros(nv); r[:n] = -pp; r[-1] = n * (1 - eps); rows.append(r); b.append(0.0)
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(b), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * nv, method="highs")
    return -res.fun if res.success else 1.0


def weighted_quantile_cutoff(scores, w_cal, w_test, alpha):
    """Oracle weighted conformal (Tibshirani et al. 2019) with known weights."""
    o = np.argsort(scores, kind="mergesort")
    s, w = np.asarray(scores, float)[o], np.asarray(w_cal, float)[o]
    tot = w.sum() + w_test
    cum = np.cumsum(w) / tot                  # mass of {s_i <= s_(k)}
    k = np.searchsorted(cum, 1.0 - alpha - 1e-12, side="left")
    return np.inf if k >= len(s) else s[k]


def split_cutoff(scores, alpha):
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]
