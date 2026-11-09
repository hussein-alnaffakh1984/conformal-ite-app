# Conformal Sensitivity ITE Explorer — code and demo

Companion repository for the manuscript *Online Conformal Sensitivity Analysis for
Individual Treatment Effects under Unmeasured Confounding and Temporal Distribution
Shift* (submitted to *Statistics and Computing*).

| Folder | What it is |
|---|---|
| `/` (root) | The interactive browser demo deployed at https://conformal-ite-app.netlify.app |
| `research/` | The Python code that produces every number, table and figure in the paper |

## The method in one paragraph

Robust weighted conformal prediction under the marginal sensitivity model (MSM)
bounds each unknown weight `w* = 1/e(x,u)` between `l(x) = 1 + (1-e)/(eΓ)` and
`h(x) = 1 + (1-e)Γ/e`. The true weights also satisfy the moment identity
`E[e(X) w*(X,U) | X, A=1] = 1`. We add its sample version, relaxed by a radius
computable from the covariates (Bernstein with a data-free variance bound), and
compute the worst case exactly by a greedy rule (checked against the
Charnes–Cooper LP). The moment-constrained cutoff is run at level `α − δ` with
`δ = 0.001`, so that it carries the same `1 − α` guarantee as the box. Online, the
adaptive level is clipped at `α − δ` so that it never narrows the robust interval.

## Reproducing the paper (`research/`)

```bash
pip install -r research/requirements.txt
# 1. data: download diabetic_data.csv (Diabetes 130-US Hospitals, Strack et al. 2014)
#    from the UCI Machine Learning Repository (dataset id 296)
python research/prep_data.py diabetic_data.csv research/diabetes_X.npz
# 2. check the exact solver against the Charnes-Cooper LP (300 instances, 900 comparisons)
python research/test_solver.py
# 3. experiments (20 replications each; outputs go to research/results/)
python research/exp_static.py main 0 20
python research/exp_static.py adversarial 0 20
python research/exp_static.py misspec 0 20
python research/exp_static.py ncal 0 20
python research/exp_static.py multi 0 20
for s in none dispersion_jump growing_dispersion neg_mean_shift; do
  python research/exp_online.py $s 0 20 500; done
python research/exp_online.py none 0 20 1000
python research/exp_online.py dispersion_jump 0 20 1000
# 4. tables and figures of the paper (written to research/paper/tables and research/paper/figures)
python research/exp_overlap.py        # gain against overlap (Table 8), ~15 s
python research/make_tables.py
python research/make_figures.py
# 5. optional: check that the demo's JavaScript gives the same cutoffs as cssa.py
python research/make_port_cases.py && node research/js_check/check_port.js
```

`results/` already holds the JSON outputs used in the paper, so step 4 can be run
without step 3. Every random draw is seeded by the replication index; with the
pinned package versions, rerunning a replication reproduces the stored values
exactly.

| File | Content |
|---|---|
| `cssa.py` | MSM box, moment constraint, exact greedy worst case, LP reference, radii |
| `dgp.py` | Semi-synthetic generators on the real covariates (MSM calibrated exactly; near-adversarial binary-U design) |
| `exp_static.py` | Static benchmark, δ sensitivity, near-adversarial design, misspecified Γ, calibration size, mechanisms × learners, ITE |
| `exp_online.py` | Online designs (fixed level, unclipped ACI, clipped ACI, standard ACI) |
| `exp_overlap.py` | Gain of the moment constraint against propensity range and alpha (cutoff level, no outcome model) |
| `make_tables.py`, `make_figures.py` | All tables and figures, from the JSON results |

## The demo (root)

Static site; no build step. The demo runs the same cutoff algorithm in JavaScript
(`cssa.js`; step 5 above checks 200 cases against `cssa.py`) on a 5,000-row random
sample of the real data. Simplifications: k-nearest-neighbour outcome model, a
logistic propensity fitted to the recorded treatment and used without the error
margin the theory requires, a single split, no grid rounding of the test weight. The
recorded outcome (30-day readmission) is binary, so widths move in steps. The
"Online results" tab shows results from the paper's experiments; it is not
computed in the browser.

**Research demonstration, not for clinical use.** With real data, coverage of an
individual's counterfactual outcome cannot be checked.

## Deploying

Netlify: publish directory `.`, no build command (see `netlify.toml`).
