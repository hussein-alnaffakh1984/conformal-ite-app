import os, json, numpy as np, sys, glob
ROOT = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(ROOT, "results", "online_*_W*_0_20.json"))):
    R=json.load(open(f)); print(f"\n=== {f.split('/')[-1]} (n={len(R)}) ===")
    print(f"{'design':9s} {'factual':>12s} {'cf all':>12s} {'cf transit':>12s} {'cf roll-min':>12s} {'fact transit':>12s} {'width':>6s}")
    for k in ["fixed","aci","aci_clip","aci_g1"]:
        def ms(key):
            v=np.array([r['summ'][k][key] for r in R]); return f"{v.mean():.3f}±{v.std(ddof=1)/np.sqrt(len(v)):.3f}"
        w=np.mean([r['summ'][k]['width_med'] for r in R])
        print(f"{k:9s} {ms('fact_cov'):>12s} {ms('cf_cov'):>12s} {ms('cf_cov_trans'):>12s} {ms('cf_roll_min'):>12s} {ms('fact_cov_trans'):>12s} {w:6.2f}")
