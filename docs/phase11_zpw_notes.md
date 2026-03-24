# Phase 11 `gnn2-zpw` Notes

## Scope

Start from the frozen `1874` multiview query-gated baseline and the negative
result from `gnn2-71v`. Keep routing, memory, and control frozen, keep the
best `full_locked` final-query-accuracy checkpoint selector, and test whether a
light training-time proxy loss can regularize the held-confirm gap without
over-rotating the main objective.

## Variants

- `11211`: `full_locked@0.1`
- `11212`: `full_locked@0.2`
- `11213`: `full_locked@0.1 + finalqueryheavy@0.1`
- `11214`: `full_locked@0.1 + finalqueryheavy@0.1 + longdistance@0.1`

All four variants reused the same frozen phase-10 `1874` multiview
query-gated baseline and the best phase-11 proxy selector from `11101`.

## Validation Summary

- `11211`: `overall 0.9873`, `fq_acc 0.9742`, `fq_route 0.9374`, `fq_exit 121.08`
- `11212`: `overall 0.9897`, `fq_acc 0.9791`, `fq_route 0.9364`, `fq_exit 121.03`
- `11213`: `overall 0.9912`, `fq_acc 0.9831`, `fq_route 0.9325`, `fq_exit 120.69`
- `11214`: `overall 0.9863`, `fq_acc 0.9722`, `fq_route 0.9295`, `fq_exit 120.39`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The split between variants is already visible on validation:

- the light `full_locked` regularizers (`11211`, `11212`) stayed below the
  selector-only baseline
- the mixed proxy regularizer (`11213`) fit the main validation best
- the widest regularizer (`11214`) improved proxy-fit summaries but clearly hurt
  main validation behavior

## Confirm Result For Lead Regularizer

Confirmed `11213`:

- `base 0.9917 / 0.9835 / 0.9395 / 121.20`
- `full_locked 0.6462 / 0.3097 / 0.8771 / 115.49`
- `finalquery_heavy 0.4470 / 0.3107 / 0.8801 / 115.84`
- `longdistance 0.5100 / 0.3005 / 0.8843 / 145.39`

Compared with the selector-only baseline from `11101`:

- `11101 full_locked`: `0.6475 / 0.3121 / 0.8771 / 115.49`
- `11213 full_locked`: `0.6462 / 0.3097 / 0.8771 / 115.49`

So the regularizer improved neither held-confirm final-query accuracy nor route.
It mainly changed how well the model fit the proxy benchmarks during training.

## Reproducibility

- `11213_rerun1` matched `11213` exactly on validation summary metrics.

That makes the family a reproducible negative rather than a one-seed miss.

## Conclusion

Light proxy-loss interpolation does not move the frozen `1874` held-confirm
ceiling. It can improve proxy-fit statistics, but that does not translate into
better confirmed final-query content. Taken together with `gnn2-e7s` and
`gnn2-71v`, the current boundary is sharp:

- mixing confirm-like data into training did not help
- selecting checkpoints on confirm-like proxies did not help
- lightly regularizing against confirm-like proxies did not help

The remaining gap looks more like a mismatch in objective geometry or training
trajectory than missing exposure to the proxy distributions themselves.

## Recommended Next Step

Keep the same frozen `1874` baseline and move one notch narrower again: test an
explicit agreement-style proxy regularizer or trust-region penalty that
constrains the main objective against held-confirm proxy predictions, rather
than simply interpolating extra proxy CE terms.
