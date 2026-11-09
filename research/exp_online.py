"""
Online experiment: does adapting alpha_t on FACTUAL errors erase the MSM counterfactual buffer?

Stream: real covariates in encounter_id order. Rows [0, n_train) fit mu_1 (RF) once and
are then frozen; the nominal propensity is the DGP's (oracle nominal, as in the static study).
Stream rows follow. Arm a = 1. Window = last W factual (A=1) units.

Designs (all use the moment-constrained robust cutoff at level alpha_t unless stated):
  fixed   : alpha_t = alpha_0 = alpha - delta
  aci     : alpha_{t+1} = alpha_t + gamma (alpha - err_t)
  aci_clip: alpha_{t+1} = min(alpha_0, alpha_t + gamma (alpha - err_t)) (buffer-preserving)
  aci_g1  : standard ACI with Gamma = 1 (ignorability weights 1/e), no sensitivity
err_t is observed only when A_t = 1; alpha_t is updated only then.
We record, for EVERY stream unit, whether Y_t(1) is in the interval: factual (A_t=1) and
counterfactual (A_t=0) coverage are both measurable because Y(1) is simulated.
"""
import sys, json, time
import os
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
from sklearn.ensemble import RandomForestRegressor
import dgp
from cssa import RobustCalibrator, box_bounds, radius_bernstein

ALPHA, GAMMA_LR, DELTA = 0.10, 0.005, 0.001
ALPHA0 = ALPHA - DELTA   # robust cutoff level and clipping level: guarantee 1 - ALPHA0 - DELTA = 0.90


def drift_profile(scn, T):
    t = np.arange(T) / T
    sigma, shift = np.ones(T), np.zeros(T)
    if scn == "growing_dispersion":
        sigma = np.where(t >= 0.5, 1.0 + 2.0 * (t - 0.5), 1.0)        # 1 -> 2 over the second half
    elif scn == "mean_shift":
        shift = np.where(t >= 0.5, 0.8, 0.0)
    elif scn == "dispersion_jump":
        sigma = np.where(t >= 0.5, 2.0, 1.0)
    elif scn == "neg_mean_shift":
        shift = np.where(t >= 0.5, -1.0, 0.0)
    return sigma, shift


def _roll_min(x, k):
    if len(x) < k:
        return float(np.mean(x))
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    return float(c.min())


def run(seed, scn, G_true=3.0, G=3.0, mech="directional", n_train=3000, T=6000, W=500):
    X = dgp.X_ALL
    rng = np.random.default_rng(seed)
    Gg = dgp.calibrate_G(G_true, mech)
    Xtr = X[:n_train]
    dtr = dgp.simulate(Xtr, Gg, mech, rng)
    m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=seed,
                               n_jobs=1).fit(Xtr[dtr["A"] == 1], dtr["Y1"][dtr["A"] == 1])
    # stream: a contiguous block of rows (encounter order) starting at a seed-specific offset, so that
    # the covariate drift of the ordering is not tied to the injected change point across replications
    off = n_train + 3000 * seed
    burn = int(np.ceil(3.0 * W))                                   # enough rows to fill W treated units
    Xs = X[off:off + burn + T]
    sig, sh = drift_profile(scn, T)
    sig = np.concatenate([np.ones(burn), sig]); sh = np.concatenate([np.zeros(burn), sh])
    d = dgp.simulate(Xs, Gg, mech, rng, sigma=sig, shift=sh)
    pred = m1.predict(Xs)
    p = d["e_nom"]
    res = np.abs(d["Y1"] - pred)
    designs = ["fixed", "aci", "aci_clip", "aci_g1"]
    out = {k: dict(cov=np.full(T, np.nan), fact=np.zeros(T, bool), width=np.full(T, np.nan),
                   alpha=np.full(T, np.nan)) for k in designs}
    a_t = {k: (ALPHA if k == "aci_g1" else ALPHA0) for k in designs}
    win_s, win_p = [], []
    for i in range(burn):                      # burn-in fills the window only
        if d["A"][i] == 1:
            win_s.append(res[i]); win_p.append(p[i])
    win_s, win_p = win_s[-W:], win_p[-W:]
    assert len(win_s) == W, f"window not full: {len(win_s)} < {W}"
    cal = cal1 = None; dirty = True
    for t in range(T):
        i = burn + t
        if dirty:
            s_arr, p_arr = np.array(win_s), np.array(win_p)
            eps = radius_bernstein(p_arr, G, DELTA)
            cal = RobustCalibrator(s_arr, p_arr, G, eps)
            cal1 = RobustCalibrator(s_arr, p_arr, 1.0, np.inf)
            dirty = False
        h_g = box_bounds(p[i], G)[1]; h_1 = box_bounds(p[i], 1.0)[1]
        for k in designs:
            c, hh = (cal1, h_1) if k == "aci_g1" else (cal, h_g)
            q = c.cutoff(a_t[k], hh)
            cov = res[i] <= q
            o = out[k]
            o["cov"][t] = cov; o["fact"][t] = d["A"][i] == 1
            o["width"][t] = 2 * q; o["alpha"][t] = a_t[k]
            if d["A"][i] == 1 and k != "fixed":
                err = 0.0 if cov else 1.0
                nxt = a_t[k] + GAMMA_LR * (ALPHA - err)
                a_t[k] = min(ALPHA0, nxt) if k == "aci_clip" else nxt
        if d["A"][i] == 1:
            win_s.append(res[i]); win_p.append(p[i])
            if len(win_s) > W:
                win_s.pop(0); win_p.pop(0)
            dirty = True
    summ = {}
    for k in designs:
        o = out[k]; f = o["fact"]; late = np.arange(T) >= T // 2
        fin = np.isfinite(o["width"])
        summ[k] = dict(
            fact_cov=float(o["cov"][f].mean()), cf_cov=float(o["cov"][~f].mean()),
            all_cov=float(o["cov"].mean()),
            fact_cov_late=float(o["cov"][f & late].mean()), cf_cov_late=float(o["cov"][~f & late].mean()),
            width_med=float(np.median(o["width"][fin])), frac_inf=float(1 - fin.mean()),
            cf_cov_trans=float(o["cov"][~f & (np.arange(T) >= T // 2) & (np.arange(T) < T // 2 + 1500)].mean()),
            fact_cov_trans=float(o["cov"][f & (np.arange(T) >= T // 2) & (np.arange(T) < T // 2 + 1500)].mean()),
            cf_roll_min=float(_roll_min(o["cov"][~f].astype(float), 400)))
    traces = {k: dict(cov=out[k]["cov"].astype(int).tolist(), fact=out[k]["fact"].astype(int).tolist(),
                      width=[float(x) if np.isfinite(x) else None for x in out[k]["width"]],
                      alpha=out[k]["alpha"].tolist()) for k in designs}
    return summ, traces


if __name__ == "__main__":
    dgp.load()
    scn = sys.argv[1]; seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    allres = []
    for sd in seeds:
        t0 = time.time()
        summ, tr = run(sd, scn, W=int(sys.argv[4]) if len(sys.argv) > 4 else 500)
        allres.append(dict(seed=sd, summ=summ, traces=tr if sd == seeds.start else None))
        print(scn, sd, {k: (round(v['fact_cov'], 3), round(v['cf_cov'], 3), round(v['width_med'], 2))
                        for k, v in summ.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(allres, open(os.path.join(ROOT, "results", ) + f"/online_{scn}_W{sys.argv[4] if len(sys.argv) > 4 else 500}_{seeds.start}_{seeds.stop}.json", "w"))
