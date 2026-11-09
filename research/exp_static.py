"""
Static (i.i.d.) experiments on the real covariates. One configuration for every table/figure:
  rows sampled without replacement from the 101,763 encounters;
  n_train = 3000 (fit mu_1, mu_0 on each arm), n_cal = 3000, n_test = 2000;
  alpha = 0.10, delta = 0.001 (MC run at level alpha - delta, box at alpha: matched 0.90 guarantee); nominal propensity = DGP nominal e(x) (oracle nominal)
  unless stated; target = coverage of Y_{n+1}(1) over the full population.
Methods (arm 1):
  naive   : unweighted split conformal on treated residuals
  wcp_g1  : weighted conformal with weights 1/e(x) (ignorability; Lei & Candes 2021)
  box     : MSM box worst case (Jin et al. 2023 style), no moment constraint
  mc_bern : MSM box + moment band, computable Bernstein radius (ours)
  mc_hoef : MSM box + moment band, Hoeffding radius
  oracle  : weighted conformal with the TRUE weights 1/e(x,u) (infeasible reference)
"""
import sys, json, time, itertools
import os
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import dgp
from cssa import (RobustCalibrator, box_bounds, radius_bernstein, radius_hoeffding,
                  split_cutoff)

ALPHA = 0.10            # nominal miscoverage for every method
DELTA = 0.001           # moment-band failure probability (fixed in advance; see paper)
ALPHA_MC = ALPHA - DELTA  # MC cutoff level, so that its guarantee 1 - ALPHA_MC - DELTA = 0.90 matches the box


def learner(kind, seed):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=seed, n_jobs=1)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)


def oracle_cutoffs(scores, w_cal, w_test, alpha):
    o = np.argsort(scores, kind="mergesort"); s = scores[o]; w = w_cal[o]
    cum = np.cumsum(w)
    need = (1 - alpha) * (cum[-1] + w_test)            # mass of {s <= q} must reach this
    k = np.searchsorted(cum, need - 1e-12, side="left")
    q = np.full(len(w_test), np.inf); ok = k < len(s); q[ok] = s[k[ok]]
    return q


def arm_cutoffs(method, sc, p_cal, p_test, G, alpha, eps_override=None, delta=None):
    if method == "naive":
        return np.full(len(p_test), split_cutoff(sc, alpha))
    if method == "wcp_g1":
        cal = RobustCalibrator(sc, p_cal, 1.0, np.inf)
        return cal.cutoffs_for_tests(alpha, box_bounds(p_test, 1.0)[1])
    d_eff = (DELTA if delta is None else delta) * (alpha / ALPHA)   # ITE: budget split across arms
    if method == "box":
        eps = np.inf
    elif method == "mc_bern":
        eps = radius_bernstein(p_cal, G, d_eff)
    elif method == "mc_hoef":
        eps = radius_hoeffding(p_cal, G, d_eff)
    elif method == "mc_eps":
        eps = eps_override
    else:
        raise ValueError(method)
    cal = RobustCalibrator(sc, p_cal, G, eps)
    lev = alpha if method == "box" else alpha - d_eff      # matched guarantee 1 - alpha
    return cal.cutoffs_for_tests(lev, box_bounds(p_test, G)[1])


def one_seed(seed, G_true, mech="directional", lrn="rf", gammas=None, n_cal=3000,
             n_train=3000, n_test=2000, methods=("naive", "wcp_g1", "box", "mc_bern", "mc_hoef"),
             do_ite=False, do_est=False, eps_grid=None, delta_grid=None):
    rng = np.random.default_rng(seed)
    gammas = gammas or [G_true]
    N = n_train + n_cal + n_test
    idx = rng.choice(len(dgp.X_ALL), N, replace=False)
    X = dgp.X_ALL[idx]
    if mech == "adversarial":
        d = dgp.simulate_adversarial(X, G_true, rng)
    else:
        d = dgp.simulate(X, dgp.calibrate_G(G_true, mech), mech, rng)
    tr, ca, te = np.arange(n_train), np.arange(n_train, n_train + n_cal), np.arange(n_train + n_cal, N)
    A, Y1, Y0, en, et = d["A"], d["Y1"], d["Y0"], d["e_nom"], d["e_true"]
    m1 = learner(lrn, seed).fit(X[tr][A[tr] == 1], Y1[tr][A[tr] == 1])
    m0 = learner(lrn, seed + 1).fit(X[tr][A[tr] == 0], Y0[tr][A[tr] == 0])
    c1 = ca[A[ca] == 1]; c0 = ca[A[ca] == 0]
    sc1 = np.abs(Y1[c1] - m1.predict(X[c1])); sc0 = np.abs(Y0[c0] - m0.predict(X[c0]))
    f1, f0 = m1.predict(X[te]), m0.predict(X[te])
    r1 = np.abs(Y1[te] - f1); ctrl = A[te] == 0
    rec = dict(seed=seed, G_true=G_true, mech=mech, lrn=lrn, n_cal1=int(len(c1)), rows=[])

    def add(method, G, q, extra=None):
        fin = np.isfinite(q)
        row = dict(method=method, G=G, cov=float(np.mean(r1 <= q)), cov_ctrl=float(np.mean(r1[ctrl] <= q[ctrl])),
                   width=float(np.mean(2 * q[fin])) if fin.any() else None, frac_inf=float(1 - fin.mean()))
        if extra: row.update(extra)
        rec["rows"].append(row)

    for G in gammas:
        for mth in methods:
            if mth in ("naive", "wcp_g1") and G != gammas[0]:
                continue
            q = arm_cutoffs(mth, sc1, en[c1], en[te], G, ALPHA)
            add(mth, 1.0 if mth in ("naive", "wcp_g1") else G, q,
                dict(eps=float(radius_bernstein(en[c1], G, DELTA)) if mth == "mc_bern" else
                     (float(radius_hoeffding(en[c1], G, DELTA)) if mth == "mc_hoef" else None)))
    # oracle (true weights) at the true model
    q = oracle_cutoffs(sc1, 1.0 / et[c1], 1.0 / et[te], ALPHA)
    add("oracle", G_true, q)
    if delta_grid is not None:
        for dl in delta_grid:
            q = arm_cutoffs("mc_bern", sc1, en[c1], en[te], gammas[0], ALPHA, delta=dl)
            add("mc_delta", gammas[0], q, dict(delta=float(dl), eps=float(radius_bernstein(en[c1], gammas[0], dl))))
    if eps_grid is not None:
        for e in eps_grid:
            q = arm_cutoffs("mc_eps", sc1, en[c1], en[te], gammas[0], ALPHA, eps_override=e)
            add("mc_eps", gammas[0], q, dict(eps=float(e)))
    if do_est:   # estimated (frozen) propensity from the training block
        lr = LogisticRegression(max_iter=2000).fit(X[tr], A[tr])
        eh = np.clip(lr.predict_proba(X)[:, 1], 0.02, 0.98)
        for mth in ("box", "mc_bern"):
            q = arm_cutoffs(mth, sc1, eh[c1], eh[te], gammas[0], ALPHA)
            add(mth + "_esthat", gammas[0], q)
    if do_ite:   # two-arm ITE intervals, each arm at alpha/2
        ite = Y1[te] - Y0[te]; cen = f1 - f0
        for mth in ("box", "mc_bern"):
            G = gammas[0]
            q1 = arm_cutoffs(mth, sc1, en[c1], en[te], G, ALPHA / 2)
            q0 = arm_cutoffs(mth, sc0, 1 - en[c0], 1 - en[te], G, ALPHA / 2)
            half = q1 + q0; fin = np.isfinite(half)
            rec["rows"].append(dict(method="ite_" + mth, G=G, cov=float(np.mean(np.abs(ite - cen) <= half)),
                                    width=float(np.mean(2 * half[fin])), frac_inf=float(1 - fin.mean())))
        # heuristic DR pseudo-outcome, plain split conformal (no confounding adjustment)
        lr = LogisticRegression(max_iter=2000).fit(X[tr], A[tr])
        eh = np.clip(lr.predict_proba(X)[:, 1], 0.02, 0.98)
        Yo = d["Y"]

        def pseudo(ix):
            a, y, e_ = A[ix], Yo[ix], eh[ix]
            p1, p0 = m1.predict(X[ix]), m0.predict(X[ix])
            return p1 - p0 + a * (y - p1) / e_ - (1 - a) * (y - p0) / (1 - e_)
        tau = learner(lrn, seed + 2).fit(X[tr], pseudo(tr))
        scp = np.abs(pseudo(ca) - tau.predict(X[ca]))
        qp = split_cutoff(scp, ALPHA)
        rec["rows"].append(dict(method="ite_pseudo_heuristic", G=1.0,
                                cov=float(np.mean(np.abs(ite - tau.predict(X[te])) <= qp)),
                                width=float(2 * qp), frac_inf=0.0))
    return rec


def main(job, s0, s1):
    dgp.load()
    out = []
    for sd in range(s0, s1):
        t0 = time.time()
        if job == "main":
            for Gt in (3.0, 5.0, 8.0):
                out.append(one_seed(sd, Gt, do_ite=True, do_est=True, delta_grid=[0.05, 0.02, 0.01, 0.005, 0.001, 0.0001]))
        elif job == "misspec":
            out.append(one_seed(sd, 5.0, gammas=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
                                methods=("naive", "wcp_g1", "box", "mc_bern")))
        elif job == "ncal":
            for nc in (1000, 2000, 4000, 8000):
                r = one_seed(sd, 5.0, n_cal=nc, methods=("box", "mc_bern", "mc_hoef"),
                             eps_grid=[0.0, 0.05, 0.1, 0.2, 0.4] if nc == 2000 else None)
                r["n_cal"] = nc; out.append(r)
        elif job == "adversarial":
            for Gt in (2.0, 3.0, 5.0):
                out.append(one_seed(sd, Gt, mech="adversarial", methods=("naive", "wcp_g1", "box", "mc_bern")))
        elif job == "multi":
            for mech, lrn, Gt in itertools.product(("directional", "threshold"), ("rf", "hgb"), (3.0, 5.0)):
                out.append(one_seed(sd, Gt, mech=mech, lrn=lrn, methods=("box", "mc_bern")))
        print(job, sd, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", ) + f"/static_{job}_{s0}_{s1}.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
