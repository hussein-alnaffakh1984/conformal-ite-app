"""All figures are drawn from the same JSON result files that produce the tables."""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
import json, collections, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "paper", "figures") + os.sep
# validated categorical palette (fixed order; colour follows the method, never its rank)
C = {"mc_bern": "#2a78d6", "box": "#eb6834", "oracle": "#1baf7a", "naive": "#eda100", "wcp_g1": "#e87ba4"}
MK = {"mc_bern": "o", "box": "s", "oracle": "^", "naive": "D", "wcp_g1": "v"}
LS = {"mc_bern": "-", "box": "--", "oracle": ":", "naive": "-.", "wcp_g1": (0, (3, 1, 1, 1))}
LAB = {"mc_bern": "Moment-constrained (ours)", "box": "MSM box", "oracle": "Oracle weights",
       "naive": "Split conformal", "wcp_g1": "Weighted, $\\Gamma=1$"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
TW = 4.68   # svjour3 smallextended text width (inches): figures are drawn at their printed size
FS = 8      # minimum lettering size in points (Springer artwork guideline)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                     "ytick.labelsize": FS, "legend.fontsize": FS, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "lines.markersize": 5, "savefig.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
R = os.path.join(ROOT, "results") + os.sep


def mse(v):
    v = np.asarray(v, float); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def target_line(ax):
    ax.axhline(0.9, color=INK, lw=0.8, ls="-", zorder=0.5)


# ---------------- Fig 1: static benchmark (coverage and width vs Gamma*)
def fig_static():
    recs = json.load(open(R + "static_main_0_20.json"))
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        for row in r["rows"]:
            D[row["method"]][r["G_true"]].append((row["cov"], row["width"]))
    Gs = [3.0, 5.0, 8.0]
    fig, ax = plt.subplots(1, 2, figsize=(TW, 2.9))
    for m in ["naive", "wcp_g1", "oracle", "box", "mc_bern"]:
        cov = [mse([x[0] for x in D[m][g]]) for g in Gs]
        ax[0].errorbar(Gs, [c[0] for c in cov], yerr=[1.96 * c[1] for c in cov], color=C[m], marker=MK[m],
                       ls=LS[m], capsize=3, label=LAB[m])
        if m in ("oracle", "box", "mc_bern"):
            w = [mse([x[1] for x in D[m][g]]) for g in Gs]
            ax[1].errorbar(Gs, [c[0] for c in w], yerr=[1.96 * c[1] for c in w], color=C[m], marker=MK[m],
                           ls=LS[m], capsize=3, label=LAB[m])
    ax[0].set(xlabel="$\\Gamma^*$ (= analyst $\\Gamma$)", ylabel="coverage of $Y(1)$",
              xticks=Gs, ylim=(0.78, 1.0)); target_line(ax[0])
    ax[1].set(xlabel="$\\Gamma^*$ (= analyst $\\Gamma$)", ylabel="mean interval width", xticks=Gs)
    ax[0].set_title("(a) validity", loc="left"); ax[1].set_title("(b) efficiency", loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(OUT + "fig_static.pdf", bbox_inches="tight")
    fig.savefig(OUT + "fig_static.png", bbox_inches="tight"); plt.close(fig)


# ---------------- Fig 2: misspecified Gamma
def fig_misspec():
    recs = json.load(open(R + "static_misspec_0_20.json"))
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        for row in r["rows"]:
            D[row["method"]][row["G"]].append((row["cov"], row["width"]))
    fig, ax = plt.subplots(1, 2, figsize=(TW, 2.8))
    for m in ["box", "mc_bern"]:
        Gs = sorted(D[m]); cov = [mse([x[0] for x in D[m][g]]) for g in Gs]; w = [mse([x[1] for x in D[m][g]]) for g in Gs]
        ax[0].errorbar(Gs, [c[0] for c in cov], yerr=[1.96 * c[1] for c in cov], color=C[m], marker=MK[m], ls=LS[m], capsize=3, label=LAB[m])
        ax[1].errorbar(Gs, [c[0] for c in w], yerr=[1.96 * c[1] for c in w], color=C[m], marker=MK[m], ls=LS[m], capsize=3, label=LAB[m])
    for m in ["naive"]:
        g = list(D[m])[0]; c = mse([x[0] for x in D[m][g]])[0]
        ax[0].axhline(c, color=C[m], ls=LS[m], lw=1.5, label=LAB[m])
    for a in ax:
        a.axvline(5.0, color=MUTED, lw=0.8); a.set_xlabel("analyst $\\Gamma$ (true $\\Gamma^*=5$)")
    ax[0].text(5.25, 0.795, "$\\Gamma=\\Gamma^*$", fontsize=FS, color=MUTED)
    ax[0].set(ylabel="coverage of $Y(1)$", ylim=(0.78, 1.0)); target_line(ax[0]); ax[1].set(ylabel="mean interval width")
    ax[0].set_title("(a) validity", loc="left"); ax[1].set_title("(b) efficiency", loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(OUT + "fig_misspec.pdf", bbox_inches="tight")
    fig.savefig(OUT + "fig_misspec.png", bbox_inches="tight"); plt.close(fig)


# ---------------- Fig 3: online traces (one representative run; tables give 20-seed SEs)
OC = {"fixed": "#6b6a63", "aci": "#eb6834", "aci_clip": "#2a78d6", "aci_g1": "#eda100"}
OL = {"fixed": "Fixed level", "aci": "Unclipped ACI", "aci_clip": "Clipped ACI (ours)",
      "aci_g1": "Standard ACI, $\\Gamma=1$"}
OS = {"fixed": ":", "aci": "--", "aci_clip": "-", "aci_g1": "-."}


def roll(x, k=400):
    x = np.asarray(x, float); c = np.convolve(x, np.ones(k) / k, mode="valid")
    return np.arange(k - 1, len(x)), c


def fig_online():
    scen = [("none", "W500", "(a) no drift\n$W=500$"), ("dispersion_jump", "W1000", "(b) dispersion jump\n$W=1000$"),
            ("neg_mean_shift", "W500", "(c) mean shift\n$W=500$")]
    fig, ax = plt.subplots(2, 3, figsize=(TW, 4.3), sharex=True, sharey=True)
    for j, (s, w, ttl) in enumerate(scen):
        tr = json.load(open(R + f"online_{s}_{w}_0_20.json"))[0]["traces"]
        for k in ["fixed", "aci", "aci_g1", "aci_clip"]:
            cov = np.array(tr[k]["cov"]); fact = np.array(tr[k]["fact"]).astype(bool); t = np.arange(len(cov))
            for i, sel in enumerate([fact, ~fact]):
                xx, yy = roll(cov[sel]); ax[i, j].plot(t[sel][xx], yy, color=OC[k], ls=OS[k], lw=1.3, label=OL[k])
        ax[0, j].set_title(ttl, loc="left")
        for i in range(2):
            ax[i, j].set_ylim(0.7, 1.0); ax[i, j].axhline(0.9, color=INK, lw=0.8, ls="-", zorder=0.5)
            if s != "none": ax[i, j].axvline(3000, color=GRID, lw=3, zorder=0)
        ax[1, j].set_xlabel("stream index $t$"); ax[1, j].set_xticks([0, 3000, 6000]); ax[1, j].set_xticklabels(["0", "3k", "6k"])
    ax[0, 0].set_ylabel("factual\ncoverage"); ax[1, 0].set_ylabel("counterfactual\ncoverage")
    ax[0, 0].set_yticks([0.7, 0.8, 0.9, 1.0])
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=1.4); fig.savefig(OUT + "fig_online.pdf", bbox_inches="tight")
    fig.savefig(OUT + "fig_online.png", bbox_inches="tight"); plt.close(fig)


# ---------------- Fig 4: calibration size and moment-band radius
def fig_ncal():
    recs = json.load(open(R + "static_ncal_0_20.json"))
    D = collections.defaultdict(list); E = collections.defaultdict(list); NC = collections.defaultdict(list)
    for r in recs:
        rows = {}
        for row in r["rows"]:
            if row["method"] == "mc_eps": E[row["eps"]].append((row["cov"], row["width"]))
            else: rows[row["method"]] = row
        D[r["n_cal"]].append(100 * (1 - rows["mc_bern"]["width"] / rows["box"]["width"]))
        NC[r["n_cal"]].append(r["n_cal1"])
    fig, ax = plt.subplots(1, 2, figsize=(TW, 2.7))
    ns = sorted(D); g = [mse(D[n]) for n in ns]; nt = [np.mean(NC[n]) for n in ns]
    ax[0].errorbar(nt, [x[0] for x in g], yerr=[1.96 * x[1] for x in g], color=C["mc_bern"], marker="o", capsize=3)
    ax[0].set(xscale="log", xlabel="treated calibration units $n$", ylabel="width reduction (%)", ylim=(0, 10))
    ax[0].set_xticks(nt); ax[0].set_xticklabels([f"{x:.0f}" for x in nt]); ax[0].minorticks_off()
    es = sorted(E); cv = [mse([x[0] for x in E[e]]) for e in es]; wd = [mse([x[1] for x in E[e]]) for e in es]
    ax[1].errorbar(es, [x[0] for x in wd], yerr=[1.96 * x[1] for x in wd], color=C["mc_bern"], marker="o", capsize=3)
    ax[1].set(xlabel="radius $\\varepsilon$ (fixed)", ylabel="mean interval width")
    ax[0].set_title("(a) gain vs calibration size", loc="left")
    ax[1].set_title("(b) width vs $\\varepsilon$", loc="left")
    ax[0].tick_params(axis="x", labelrotation=0)
    fig.tight_layout(); fig.savefig(OUT + "fig_ncal.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_ncal.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    fig_static(); fig_misspec(); fig_online(); fig_ncal(); print("figures written")
