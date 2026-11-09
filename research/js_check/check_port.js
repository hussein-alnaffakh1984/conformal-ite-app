const { RobustCalibrator, radiusBernstein } = require("../../cssa.js");
const cases = require("../port_cases.json"); let bad = 0, maxr = 0;
for (const c of cases) {
  const eps = c.eps === null ? Infinity : c.eps;
  const q = new RobustCalibrator(c.s, c.p, c.G, eps).cutoff(c.a, c.h);
  const qp = c.q === null ? Infinity : c.q;
  if (!(q === qp || Math.abs(q - qp) < 1e-12)) bad++;
  maxr = Math.max(maxr, Math.abs(radiusBernstein(c.p, c.G, 0.05) - c.rb));
}
console.log(`JS vs Python cutoffs: ${cases.length - bad}/${cases.length} identical; max radius diff ${maxr.toExponential(2)}`);
