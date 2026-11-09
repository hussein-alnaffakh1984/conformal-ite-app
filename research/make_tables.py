"""Generate every results table of the manuscript (LaTeX fragments) from the JSON files."""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
import json, collections, numpy as np
R = os.path.join(ROOT, "results") + os.sep; OUT = os.path.join(ROOT, "paper", "tables") + os.sep
import os; os.makedirs(OUT, exist_ok=True)


def mse(v):
    v = np.asarray(v, float); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def f(m, se, d=3):
    sse = f"{se:.{d}f}"
    if float(sse) == 0: sse = "$<$" + f"{10 ** -d:.{d}f}"
    return f"{m:.{d}f} ({sse})"


def sg(x, d=1):
    """signed number with a true minus sign"""
    t = f"{x:.{d}f}"
    return t.replace("-", "$-$")


# ---------- static benchmark
recs = json.load(open(R + "static_main_0_20.json"))
D = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    for row in r["rows"]:
        D[(r["G_true"], row["method"])]["cov"].append(row["cov"])
        D[(r["G_true"], row["method"])]["ctrl"].append(row.get("cov_ctrl", np.nan))
        D[(r["G_true"], row["method"])]["w"].append(row["width"])
        if row.get("eps") is not None: D[(r["G_true"], row["method"])]["eps"].append(row["eps"])
names = [("naive", "Split conformal (no weighting)"), ("wcp_g1", "Weighted, $\\Gamma=1$ (ignorability)"),
         ("box", "MSM box"), ("mc_hoef", "Moment-constrained, Hoeffding"),
         ("mc_bern", "Moment-constrained, Bernstein (ours)"), ("oracle", "Oracle weights (infeasible)")]
L = []
for Gt in (3.0, 5.0, 8.0):
    L.append(f"\\multicolumn{{5}}{{l}}{{$\\Gamma^*=\\Gamma={Gt:.0f}$}}\\\\")
    for m, nm in names:
        d = D[(Gt, m)]; c = mse(d["cov"]); w = mse(d["w"])
        L.append(f"\\quad {nm} & {f(*c)} & {np.min(d['cov']):.3f} & {np.nanmean(d['ctrl']):.3f} & {w[0]:.2f} ({w[1]:.2f})\\\\")
    if Gt != 8.0: L.append("\\addlinespace")
open(OUT + "tab_static_body.tex", "w").write("\n".join(L) + "\n")
gains = {Gt: 100 * (1 - mse(D[(Gt, 'mc_bern')]['w'])[0] / mse(D[(Gt, 'box')]['w'])[0]) for Gt in (3.0, 5.0, 8.0)}
eps = {Gt: np.mean(D[(Gt, 'mc_bern')]['eps']) for Gt in (3.0, 5.0, 8.0)}
epsH = {Gt: np.mean(D[(Gt, 'mc_hoef')]['eps']) for Gt in (3.0, 5.0, 8.0)}

# ---------- ITE + estimated propensity
L = []; est = []; ite_gain = []; maxse = 0
for Gt in (3.0, 5.0, 8.0):
    row = [f"{Gt:.0f}"]
    for m in ("ite_box", "ite_mc_bern", "ite_pseudo_heuristic"):
        d = D[(Gt, m)]; c = mse(d["cov"]); w = mse(d["w"]); maxse = max(maxse, c[1])
        row += [f"{c[0]:.3f}", f"{w[0]:.2f}"]
    ite_gain.append(100 * (1 - mse(D[(Gt, 'ite_mc_bern')]['w'])[0] / mse(D[(Gt, 'ite_box')]['w'])[0]))
    L.append(" & ".join(row) + "\\\\")
    de, dt = mse(D[(Gt, "mc_bern_esthat")]["cov"]), mse(D[(Gt, "mc_bern")]["cov"])
    est.append(f"{de[0]:.3f} vs {dt[0]:.3f} at $\\Gamma={Gt:.0f}$")
open(OUT + "tab_ite_body_main.tex", "w").write("\n".join(L) + "\n")
open(OUT + "tab_est_inline.tex", "w").write("; ".join(est) + " (estimated vs nominal propensity).")
print("ITE gain MC vs box (%):", [round(g, 2) for g in ite_gain], "max coverage SE", round(maxse, 4))

# ---------- multi-scenario
recs = json.load(open(R + "static_multi_0_20.json"))
M = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    rows = {x["method"]: x for x in r["rows"]}; k = (r["mech"], r["lrn"], r["G_true"])
    for m in ("box", "mc_bern", "oracle"):
        M[k][m + "c"].append(rows[m]["cov"]); M[k][m + "w"].append(rows[m]["width"])
    M[k]["g"].append(100 * (1 - rows["mc_bern"]["width"] / rows["box"]["width"]))
L = []
for k in sorted(M):
    d = M[k]; mech = "directional" if k[0] == "directional" else "$|U|$-threshold"
    lrn = "RF" if k[1] == "rf" else "HGB"
    L.append(f"{mech} & {lrn} & {k[2]:.0f} & {f(*mse(d['boxc']))} & {f(*mse(d['mc_bernc']))} & {np.min(d['mc_bernc']):.3f} & "
             f"{mse(d['boxw'])[0]:.2f} & {mse(d['mc_bernw'])[0]:.2f} & {mse(d['g'])[0]:.1f} ({mse(d['g'])[1]:.1f})\\\\")
open(OUT + "tab_multi_body.tex", "w").write("\n".join(L) + "\n")
mgains = [mse(M[k]["g"])[0] for k in M]

# ---------- online
L = []
scen = [("none", "W500", "No drift ($W=500$)"), ("dispersion_jump", "W500", "Dispersion jump ($W=500$)"),
        ("growing_dispersion", "W500", "Growing dispersion ($W=500$)"), ("neg_mean_shift", "W500", "Mean shift ($W=500$)"),
        ("none", "W1000", "No drift ($W=1000$)"), ("dispersion_jump", "W1000", "Dispersion jump ($W=1000$)")]
dn = [("fixed", "Fixed level"), ("aci", "Unclipped ACI"), ("aci_clip", "Clipped ACI (ours)"), ("aci_g1", "Standard ACI, $\\Gamma=1$")]
ON = {}
for s, w, lab in scen:
    Rr = json.load(open(R + f"online_{s}_{w}_0_20.json"))
    L.append(f"\\multicolumn{{7}}{{l}}{{{lab}}}\\\\")
    for k, nm in dn:
        g = lambda key: mse([r["summ"][k][key] for r in Rr])
        fc, cc, ct, rm = g("fact_cov"), g("cf_cov"), g("cf_cov_trans"), g("cf_roll_min")
        wd = np.mean([r["summ"][k]["width_med"] for r in Rr])
        ret = 1 - max(0.0, 0.9 - ct[0]) / 0.9
        ON[(s, w, k)] = dict(fc=fc[0], cc=cc[0], ct=ct[0], rm=rm[0], wd=wd, ret=ret)
        tr = f"{ct[0]:.3f}"
        retc = f"{ret:.3f}"
        L.append(f"\\quad {nm} & {f(*fc)} & {f(*cc)} & {tr} & {rm[0]:.3f} & {retc} & {wd:.2f}\\\\")
    if (s, w) != scen[-1][:2]: L.append("\\addlinespace")
open(OUT + "tab_online_body.tex", "w").write("\n".join(L) + "\n")

# ---------- calibration size
recs = json.load(open(R + "static_ncal_0_20.json"))
NC = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    rows = {x["method"]: x for x in r["rows"] if x["method"] != "mc_eps"}
    d = NC[r["n_cal"]]; d["n1"].append(r["n_cal1"]); d["eb"].append(rows["mc_bern"]["eps"]); d["eh"].append(rows["mc_hoef"]["eps"])
    d["c"].append(rows["mc_bern"]["cov"]); d["g"].append(100 * (1 - rows["mc_bern"]["width"] / rows["box"]["width"]))
    d["gh"].append(100 * (1 - rows["mc_hoef"]["width"] / rows["box"]["width"]))
L = []
for n in sorted(NC):
    d = NC[n]
    L.append(f"{np.mean(d['n1']):.0f} & {np.mean(d['eh']):.3f} & {np.mean(d['eb']):.3f} & {f(*mse(d['c']))} & {np.min(d['c']):.3f} & "
             f"{mse(d['gh'])[0]:.1f} & {mse(d['g'])[0]:.1f} ({mse(d['g'])[1]:.1f})\\\\")
open(OUT + "tab_ncal_body.tex", "w").write("\n".join(L) + "\n")
ngains = [mse(NC[n]["g"])[0] for n in NC]

# ---------- delta sensitivity (main runs)
recs = json.load(open(R + "static_main_0_20.json"))
DL = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    rows = r["rows"]; box = [x for x in rows if x["method"] == "box"][0]
    for x in rows:
        if x["method"] == "mc_delta":
            k = (r["G_true"], x["delta"]); DL[k]["c"].append(x["cov"]); DL[k]["w"].append(x["width"])
            DL[k]["e"].append(x["eps"]); DL[k]["g"].append(100 * (1 - x["width"] / box["width"]))
L = []; dgain = {}
for Gt in (3.0, 5.0, 8.0):
    for dl in sorted({k[1] for k in DL if k[0] == Gt}, reverse=True):
        d = DL[(Gt, dl)]; g = mse(d["g"]); dgain[(Gt, dl)] = g[0]
        L.append(f"{Gt:.0f} & {dl:g} & {0.1 - dl:g} & {np.mean(d['e']):.3f} & {f(*mse(d['c']))} & {np.min(d['c']):.3f} & "
                 f"{mse(d['w'])[0]:.2f} & {sg(g[0])} ({g[1]:.1f})\\\\")
    if Gt != 8.0: L.append("\\addlinespace")
open(OUT + "tab_delta_body.tex", "w").write("\n".join(L) + "\n")

# ---------- near-adversarial generator
recs = json.load(open(R + "static_adversarial_0_20.json"))
AD = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    rows = {x["method"]: x for x in r["rows"]}; k = r["G_true"]
    for m in ("naive", "wcp_g1", "box", "mc_bern", "oracle"):
        AD[k][m + "c"].append(rows[m]["cov"]); AD[k][m + "w"].append(rows[m]["width"])
    AD[k]["g"].append(100 * (1 - rows["mc_bern"]["width"] / rows["box"]["width"]))
L = []; agains = []
for k in sorted(AD):
    d = AD[k]; agains.append(mse(d["g"])[0])
    L.append(f"{k:.0f} & {mse(d['naivec'])[0]:.3f} & {f(*mse(d['boxc']))} & {f(*mse(d['mc_bernc']))} & {np.min(d['mc_bernc']):.3f} & "
             f"{np.min(d['boxc']):.3f} & {mse(d['boxw'])[0]:.2f} & {mse(d['mc_bernw'])[0]:.2f} & {mse(d['g'])[0]:.1f} ({mse(d['g'])[1]:.1f}) & {mse(d['oraclec'])[0]:.3f}\\\\")
open(OUT + "tab_adv_body.tex", "w").write("\n".join(L) + "\n")

allg = list(gains.values()) + mgains + ngains + agains
# ---------- gain against overlap (cutoff-level study, no outcome model)
OV = json.load(open(R + "overlap_gain.json"))
L = []
for o in OV:
    lo, hi = o["range"]
    L.append(f"{{[{lo:.2f}, {hi:.2f}]}} & {o['G']:.0f} & {o['alpha']:.2f} & {o['gain'][0]:.1f} ({o['gain'][1]:.2f})\\\\")
open(OUT + "tab_overlap_body.tex", "w").write("\n".join(L) + "\n")

summary = dict(main_gain=gains, eps_bern=eps, eps_hoef=epsH, gain_range_all=(min(allg), max(allg)),
               delta_gain={f"{k[0]}|{k[1]}": v for k, v in dgain.items()}, adv_gain=agains, multi_gain=mgains, ncal_gain=ngains,
               online={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in ON.items()})
json.dump(summary, open(OUT + "numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "online"}, indent=1, default=float))
for k, v in ON.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
