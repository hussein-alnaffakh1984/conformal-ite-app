"""
Semi-synthetic data on the real Diabetes-130 covariates (standardized, encounter order).

Hidden confounder z ~ N(0,1), U = z * Us(x).
  Y(1) = mu(x) + U + 0.5 xi1,     Y(0) = mu(x) - 0.5 - U + 0.5 xi0     (ITE = 0.5 + 2U + noise)
Treatment: odds(e(x,z)) = odds(e0(x)) / lambda(z),
  directional : lambda(z) = G_gen ** (-tanh(c z))          (high z -> treated -> high Y(1))
  threshold   : lambda(z) = G_gen ** (-tanh(c (|z| - 1)))   (|z| large -> treated; symmetric in z)
Nominal propensity e(x) = E_z[e(x,z)] (Gauss-Hermite, 60 nodes).
The MSM holds w.r.t. the nominal e(x) with Gamma*(x) = sup_z OR(e(x,z), e(x)); G_gen is
calibrated so that sup over ALL 101,763 covariate rows of Gamma*(x) equals the target Gamma*.
"""
import os
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
from scipy.special import expit, logit

X_ALL = None
NAMES = None
_GH_z, _GH_w = np.polynomial.hermite_e.hermegauss(60)
_GH_w = _GH_w / _GH_w.sum()
C_SLOPE = 2.0
_ZGRID = np.linspace(-8, 8, 801)


def load(path=os.path.join(ROOT, "diabetes_X.npz")):
    global X_ALL, NAMES
    d = np.load(path, allow_pickle=True)
    X_ALL, NAMES = d["X"], list(d["names"])
    return X_ALL


def col(X, name):
    return X[:, NAMES.index(name)]


def e0(X):
    """Baseline propensity, kept in [0.08, 0.92] (overlap)."""
    lin = (-0.15 + 0.6 * np.tanh(col(X, "num_medications")) + 0.5 * np.tanh(col(X, "number_inpatient"))
           - 0.4 * np.tanh(col(X, "age_mid")) + 0.4 * np.tanh(col(X, "time_in_hospital")))
    return 0.08 + 0.84 * expit(lin)


def mu(X):
    return (1.0 + 0.6 * col(X, "age_mid") + 0.5 * np.tanh(col(X, "number_inpatient"))
            - 0.4 * col(X, "time_in_hospital") + 0.5 * np.sin(col(X, "num_medications"))
            + 0.3 * np.tanh(col(X, "number_emergency")))


def us(X):
    return 1.0 + 0.5 * np.sin(2.5 * col(X, "num_lab_procedures"))


def lam(z, G, mech):
    if mech == "directional":
        return G ** (-np.tanh(C_SLOPE * z))
    if mech == "threshold":
        return G ** (-np.tanh(C_SLOPE * (np.abs(z) - 1.0)))
    raise ValueError(mech)


def e_xz(e0x, z, G, mech):
    o = e0x / (1 - e0x) / lam(z, G, mech)
    return o / (1 + o)


def e_nominal(e0x, G, mech):
    return (e_xz(e0x[:, None], _GH_z[None, :], G, mech) * _GH_w[None, :]).sum(1)


def gamma_star(e0x, G, mech):
    """sup_z OR(e(x,z), e(x)) for each x (grid over z in [-8, 8])."""
    en = e_nominal(e0x, G, mech)
    ez = e_xz(e0x[:, None], _ZGRID[None, :], G, mech)
    lor = np.abs(logit(ez) - logit(en)[:, None]).max(1)
    return np.exp(lor)


_CAL = {}


def calibrate_G(target, mech):
    """Gamma*(x) depends on x only through e0(x): take the sup over a fine grid spanning
    the population range of e0 (end points included exactly)."""
    key = (round(target, 6), mech)
    if key in _CAL:
        return _CAL[key]
    e_pop = e0(X_ALL)
    e0x = np.linspace(e_pop.min(), e_pop.max(), 2001)
    lo, hi = 1.0, target
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if gamma_star(e0x, m, mech).max() > target:
            hi = m
        else:
            lo = m
    _CAL[key] = lo
    return lo


def simulate(X, G_gen, mech, rng, sigma=1.0, shift=0.0):
    """sigma, shift may be arrays (per-row drift)."""
    n = len(X)
    e0x = e0(X)
    z = rng.standard_normal(n)
    ez = e_xz(e0x, z, G_gen, mech)
    A = rng.binomial(1, ez)
    U = z * us(X)
    m = mu(X) + shift
    Y1 = m + sigma * (U + 0.5 * rng.standard_normal(n))
    Y0 = m - 0.5 + sigma * (-U + 0.5 * rng.standard_normal(n))
    en = e_nominal(e0x, G_gen, mech)
    return dict(A=A, Y1=Y1, Y0=Y0, e_true=ez, e_nom=en, Y=np.where(A == 1, Y1, Y0))


def simulate_adversarial(X, G, rng):
    """Near worst case: binary U with OR{e(x,u), e(x)} = G^{2u-1} exactly, so every unit's weight
    1/e(x,u) sits at l(x) (u=1) or h(x) (u=0), and U is aligned with the sign of the residual:
    treated units are mostly u=1 (positive deviations), controls mostly u=0 (negative).
    Nominal propensity e(x) = e0(x) holds exactly with P(U=1|x) = (e - e_0u)/(e_1u - e_0u)."""
    n = len(X); e = e0(X); o = e / (1 - e)
    e1 = o * G / (1 + o * G); e0u = (o / G) / (1 + o / G)
    q = (e - e0u) / (e1 - e0u)
    U = rng.binomial(1, q)
    ez = np.where(U == 1, e1, e0u)
    A = rng.binomial(1, ez)
    sgn = 2 * U - 1
    dev = np.abs(rng.standard_normal(n)) * us(X)
    Y1 = mu(X) + sgn * dev + 0.2 * rng.standard_normal(n)
    Y0 = mu(X) - 0.5 - sgn * dev + 0.2 * rng.standard_normal(n)
    return dict(A=A, Y1=Y1, Y0=Y0, e_true=ez, e_nom=e, Y=np.where(A == 1, Y1, Y0))
