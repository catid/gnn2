# Phase 11 `gnn2-vok` Notes

## Scope

Start from the same frozen `1874` multiview query-gated baseline used in
`gnn2-e7s`, `gnn2-71v`, `gnn2-zpw`, and `gnn2-m5v`. Keep routing, memory,
control, and the frozen reader path unchanged. Stop using confirm-like proxy
objectives and instead test direct final-query prediction shaping on the main
train split: focal-style losses and margin-style losses applied only on
`needs_final_query` examples.

## Variants

- `11411`: focal `gamma=1.0`, weight `0.25`
- `11412`: focal `gamma=2.0`, weight `0.50`
- `11413`: margin `0.2`, weight `0.25`
- `11414`: margin `0.4`, weight `0.50`

All four variants reused the same frozen phase-10 `1874` multiview
query-gated checkpoint and the same `full_locked` final-query-accuracy
selection rule from `11101`.

## Validation Summary

- `11101` selector-only baseline: `overall 0.9937`, `fq_acc 0.9871`,
  `fq_route 0.9533`, `fq_exit 122.48`
- `11411`: `overall 0.9917`, `fq_acc 0.9831`, `fq_route 0.9474`,
  `fq_exit 121.93`
- `11412`: `overall 0.9917`, `fq_acc 0.9831`, `fq_route 0.9434`,
  `fq_exit 121.64`
- `11413`: `overall 0.9844`, `fq_acc 0.9682`, `fq_route 0.9404`,
  `fq_exit 121.39`
- `11414`: `overall 0.9917`, `fq_acc 0.9831`, `fq_route 0.9325`,
  `fq_exit 120.49`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The best shaping variant was `11411`, but it was still strictly below the
selector-only baseline on every main validation metric. The stronger focal and
margin settings increased the shaping term further but did not reveal a better
trajectory.

## Reproducibility

- `11411_rerun1` matched `11411` exactly on validation summary metrics.

That makes the family a reproducible negative rather than a one-seed miss.

## Conclusion

Direct final-query margin shaping also fails on frozen `1874`. Relative to the
selector-only baseline:

- focal shaping weakens validation fit while adding a non-trivial shaping loss
- stronger focal weighting weakens route further
- margin shaping is worse than focal shaping
- none of the four variants improves the held-confirm candidate trajectory

Taken together with the earlier phase-11 negatives, the current boundary is now
sharp:

- confirm-aware data mixing did not help
- proxy-based checkpoint selection did not help
- light proxy CE interpolation did not help
- proxy-agreement KL did not help
- direct final-query focal / margin shaping did not help

The remaining gap no longer looks like a simple objective-weighting problem on
the existing frozen strong-source baseline.

## Recommended Next Step

Stop iterating on narrow scalar objective tweaks for frozen `1874` and move to
a different output-geometry axis: a prototype or metric-style answer head on
the same frozen source, still without reopening routing, memory, or control.
