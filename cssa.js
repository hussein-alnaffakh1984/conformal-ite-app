// cssa.js -- browser port of cssa.py (moment-constrained MSM robust cutoff).
// Kept identical in logic to cssa.RobustCalibrator; checked against Python by check_port.js.
function boxBounds(p, G) {
  return [1 + (1 - p) / (p * G), 1 + (1 - p) * G / p];
}
function radiusBernstein(ps, G, delta) {
  let V = 0, b = 0;
  for (const p of ps) {
    const [lo, hi] = boxBounds(p, G); const m = p * lo, M = p * hi;
    V += Math.max(M - 1, 0) * Math.max(1 - m, 0); b = Math.max(b, M - 1, 1 - m, 0);
  }
  const L = Math.log(2 / delta);
  const t = b * L / 3 + Math.sqrt((b * L / 3) ** 2 + 2 * L * V);
  return t / ps.length;
}
class RobustCalibrator {
  constructor(scores, ps, G, eps) {
    const o = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b] || a - b);
    this.s = o.map(i => scores[i]); this.p = o.map(i => ps[i]); this.n = this.s.length;
    this.lo = this.p.map(v => boxBounds(v, G)[0]); this.hi = this.p.map(v => boxBounds(v, G)[1]);
    this.ordA = this.p.map((_, i) => i).sort((a, b) => this.p[b] - this.p[a] || a - b);
    this.Lb = isFinite(eps) ? this.n * (1 - eps) : -Infinity;
    this.Ub = isFinite(eps) ? this.n * (1 + eps) : Infinity;
  }
  start(k) { // first index with s > s_(k), k 1-based
    const v = this.s[k - 1]; let lo = 0, hi = this.n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.s[m] <= v) lo = m + 1; else hi = m; }
    return lo;
  }
  tailMass(k, hTest) {
    const st = this.start(k), n = this.n, w = new Array(n);
    let S = 0;
    for (let i = 0; i < n; i++) { w[i] = i >= st ? this.hi[i] : this.lo[i]; S += this.p[i] * w[i]; }
    if (S > this.Ub) {
      let need = S - this.Ub;
      for (const i of this.ordA) {
        if (i < st) continue;
        const cap = (this.hi[i] - this.lo[i]) * this.p[i];
        if (cap >= need) { w[i] -= need / this.p[i]; need = 0; break; }
        w[i] = this.lo[i]; need -= cap;
      }
      if (need > 1e-12) return 1.0;
    } else if (S < this.Lb) {
      let need = this.Lb - S;
      for (const i of this.ordA) {
        if (i >= st) continue;
        const cap = (this.hi[i] - this.lo[i]) * this.p[i];
        if (cap >= need) { w[i] += need / this.p[i]; need = 0; break; }
        w[i] = this.hi[i]; need -= cap;
      }
      if (need > 1e-12) return 1.0;
    }
    let num = hTest, den = hTest;
    for (let i = 0; i < n; i++) { den += w[i]; if (i >= st) num += w[i]; }
    return num / den;
  }
  cutoff(alpha, hTest) {
    if (alpha <= 0) return Infinity;
    const n = this.n;
    if (this.tailMass(n, hTest) > alpha) return Infinity;
    if (this.tailMass(1, hTest) <= alpha) return this.s[0];
    let lo = 1, hi = n;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (this.tailMass(m, hTest) <= alpha) hi = m; else lo = m; }
    return this.s[hi - 1];
  }
}
if (typeof module !== "undefined") module.exports = { boxBounds, radiusBernstein, RobustCalibrator };
