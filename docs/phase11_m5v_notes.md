# Phase 11 `gnn2-m5v` Notes

## Scope

Start from the frozen `1874` multiview query-gated baseline and the negative
results from `gnn2-e7s`, `gnn2-71v`, and `gnn2-zpw`. Keep routing, memory, and
control frozen. Keep the best `full_locked` final-query-accuracy selector, but
replace proxy CE interpolation with a tighter agreement-style proxy regularizer:
logit KL from the current student to the frozen source model on confirm-like
auxiliary batches, restricted to `final_query_only` examples.

## Variants

- `11311`: `full_locked` agreement KL at `0.05`
- `11312`: `full_locked` agreement KL at `0.10`
- `11313`: `full_locked@0.05 + finalqueryheavy@0.05`
- `11314`: `full_locked@0.05 + finalqueryheavy@0.05 + longdistance@0.05`

All four variants reused the same frozen phase-10 `1874` multiview
query-gated baseline and the same frozen-source teacher checkpoint from
`10012`.

## Validation Summary

- `11311`: `overall 0.9873`, `fq_acc 0.9742`, `fq_route 0.9394`, `fq_exit 121.07`
- `11312`: `overall 0.9863`, `fq_acc 0.9722`, `fq_route 0.9275`, `fq_exit 120.40`
- `11313`: `overall 0.9888`, `fq_acc 0.9772`, `fq_route 0.9444`, `fq_exit 121.99`
- `11314`: `overall 0.9854`, `fq_acc 0.9722`, `fq_route 0.9454`, `fq_exit 121.58`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The strongest variant was `11313`, but even that stayed clearly below the
selector-only baseline from `11101`:

- `11101`: `overall 0.9937`, `fq_acc 0.9871`, `fq_route 0.9533`, `fq_exit 122.48`
- `11313`: `overall 0.9888`, `fq_acc 0.9772`, `fq_route 0.9444`, `fq_exit 121.99`

## Reproducibility

- `11313_rerun1` matched `11313` exactly on validation summary metrics.

That makes the family a reproducible negative rather than a one-seed miss.

## Conclusion

Agreement-style proxy KL also failed to move the frozen `1874` ceiling. Relative
to the selector-only baseline, it weakened base validation fit, weakened
final-query accuracy, and did not reveal a better proxy-aligned training
trajectory.

Taken together, phase 11 now has a clean chain of negatives on the same frozen
strong-source baseline:

- confirm-aware data mixing did not help
- proxy-based checkpoint selection did not help
- light proxy CE interpolation did not help
- proxy-agreement KL did not help

The remaining gap no longer looks like a missing confirm-like proxy objective.
It looks more like the baseline itself has reached the limit of what these
narrow training-objective perturbations can unlock.

## Recommended Next Step

Stop pushing on confirm-like proxy objectives for this frozen baseline and move
to a different narrow axis: direct final-query prediction shaping on the main
train split, such as margin or calibration-oriented losses that target the
answer distribution itself rather than auxiliary proxy splits.
