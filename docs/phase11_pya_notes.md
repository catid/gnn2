# Phase 11 `gnn2-pya` Notes

## Scope

Start from the same frozen `1874` multiview query-gated selector baseline used
in `gnn2-71v`, `gnn2-zpw`, `gnn2-m5v`, and `gnn2-vok`. Keep routing, memory,
control, and the frozen multiview reader path fixed, but replace the standard
linear answer head with a cosine / prototype-style head. The goal is to test
whether the remaining held-confirm gap is output-geometry-limited rather than a
reader or objective-weighting problem.

## Variants

- `11511`: cosine head, no projection, initial logit scale `10.0`
- `11512`: cosine head, no projection, initial logit scale `20.0`
- `11513`: cosine head with metric projection `d=64`, initial scale `10.0`
- `11514`: cosine head with metric projection `d=64` and prototype-pull weight
  `0.1`

All four variants reused the same frozen phase-10 `1874` multiview
query-gated checkpoint and the same `full_locked` final-query-accuracy
selection rule introduced in `11101`.

## Validation Summary

- `11101` selector-only baseline: `overall 0.9937`, `fq_acc 0.9871`,
  `fq_route 0.9533`, `fq_exit 122.48`
- `11511`: `overall 0.9941`, `fq_acc 0.9881`, `fq_route 0.9345`,
  `fq_exit 120.41`
- `11512`: `overall 0.9937`, `fq_acc 0.9871`, `fq_route 0.9325`,
  `fq_exit 120.84`
- `11513`: `overall 0.9951`, `fq_acc 0.9901`, `fq_route 0.9275`,
  `fq_exit 119.89`
- `11514`: `overall 0.9951`, `fq_acc 0.9911`, `fq_route 0.9374`,
  `fq_exit 121.05`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The family pattern is consistent:

- answer accuracy improves slightly relative to the selector-only baseline
- route match gets worse in every prototype variant
- exit timing also gets earlier in every prototype variant

That makes the best prototype result look like a different tradeoff, not a real
improvement.

## Reproducibility

- `11514_rerun1` matched `11514` exactly on validation summary metrics.

That is enough to retire the family as a reproducible negative rather than a
one-seed miss.

## Conclusion

Prototype / cosine answer heads do not solve the frozen `1874` held-confirm
problem. Relative to the selector-only baseline:

- plain cosine heads do not help
- larger cosine scale does not help
- a learned metric projection increases answer fit further, but worsens route
  and exit even more
- light prototype-pull regularization stabilizes the projection head slightly,
  but still leaves it clearly below baseline on route and exit

This sharpens the phase-11 boundary again: the remaining gap does not look like
a simple scalar objective problem, and it also does not look like a simple
output-geometry problem that can be fixed by swapping the answer head for a
metric classifier.

## Recommended Next Step

Stop iterating on frozen-head output geometry in isolation. The next narrow
follow-up should test a small **mixture-of-readers / mixture-of-heads**
architecture on frozen `1874`, still without reopening routing, memory, or
control, to see whether the ceiling comes from needing multiple specialized
answer subspaces rather than one global readout geometry.
