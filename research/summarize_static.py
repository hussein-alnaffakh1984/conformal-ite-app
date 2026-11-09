import os, json, numpy as np, sys, collections
ROOT = os.path.dirname(os.path.abspath(__file__))
def agg(recs, filt=lambda r: True):
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if not filt(r): continue
        for row in r["rows"]:
            key = (r["G_true"], row["method"], row["G"])
            for m in ("cov","cov_ctrl","width","frac_inf","eps"):
                if row.get(m) is not None: D[key][m].append(row[m])
    out = {}
    for k, v in D.items():
        out[k] = {m: (np.mean(x), np.std(x, ddof=1)/np.sqrt(len(x)), np.min(x), len(x)) for m, x in v.items()}
    return out
if __name__ == "__main__":
    recs = json.load(open(os.path.join(ROOT, "results", "static_main_0_20.json")))
    A = agg(recs)
    for Gt in (3.0,5.0,8.0):
        print(f"\nGamma*={Gt}")
        for meth in ["naive","wcp_g1","box","mc_bern","mc_hoef","oracle","box_esthat","mc_bern_esthat","ite_box","ite_mc_bern","ite_pseudo_heuristic"]:
            k=[kk for kk in A if kk[0]==Gt and kk[1]==meth]
            if not k: continue
            a=A[k[0]]; c=a["cov"]; w=a.get("width",(np.nan,np.nan,np.nan,0)); cc=a.get("cov_ctrl",(np.nan,)*4); e=a.get("eps")
            print(f"  {meth:22s} cov {c[0]:.3f}±{c[1]:.3f} (min {c[2]:.3f})  ctrl {cc[0]:.3f}  width {w[0]:.2f}±{w[1]:.2f}" + (f"  eps {e[0]:.3f}" if e else ""))
        b=A[(Gt,'box',Gt)]['width'][0]; m=A[(Gt,'mc_bern',Gt)]['width'][0]; h=A[(Gt,'mc_hoef',Gt)]['width'][0]
        print(f"  -> MC(Bernstein) narrower than box by {100*(1-m/b):.1f}%; MC(Hoeffding) by {100*(1-h/b):.1f}%")
    print("n_cal treated:", np.mean([r['n_cal1'] for r in recs]))
