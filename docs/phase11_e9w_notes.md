# Phase 11 `gnn2-e9w` Notes

## Scope

`udf` ended with a clear frozen-`1201` interpolation boundary:

- `12423` was the best nearby interpolation point and reran exactly
- but `phase11_verify` showed that its held-confirm metrics were identical to
  `12412`

This follow-up kept the same frozen `1201` core and the same interpolation
mechanism, and changed only the selector / objective mix around that `12423`
boundary.

## Variants

- `12511`: proxy `full_locked` final-query-accuracy selector only
- `12512`: stronger `final_query_weight=4.0` under overall selection
- `12513`: proxy selector plus `final_query_weight=4.0`
- `12514`: proxy selector plus `final_query_weight=4.0` plus light
  `full_locked + finalqueryheavy + longdistance` confirm-mix
- `12514_rerun1`: exact rerun of the strongest scout

## Result

The family produced a real base-tradeoff improvement, but not a held-confirm
frontier move.

Selected base-test metrics:

- `12423`: `0.6344 / 0.3215 / 0.7674 / 123.75`
- `12511`: `0.6198 / 0.3095 / 0.7714 / 124.27`
- `12512`: `0.6253 / 0.3189 / 0.7868 / 124.40`
- `12513`: `0.6315 / 0.3322 / 0.7774 / 123.54`
- `12514`: `0.6423 / 0.3549 / 0.7721 / 124.62`
- `12514_rerun1`: exact match to `12514`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Confirm Comparison

`phase11_verify` on `12514`:

- `base`: `0.6325 / 0.3483 / 0.7894 / 123.65`
- `full_locked`: `0.1217 / 0.2500 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1937 / 0.2443 / 0.9885 / 126.36`
- `longdistance`: `0.1820 / 0.2571 / 0.9922 / 158.65`

For comparison `12423` verify remained:

- `base`: `0.6423 / 0.3282 / 0.7848 / 123.68`
- `full_locked`: `0.1217 / 0.2500 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1937 / 0.2443 / 0.9885 / 126.36`
- `longdistance`: `0.1820 / 0.2571 / 0.9922 / 158.65`

## Interpretation

- `12514` is not a one-seed artifact:
  the exact rerun matched bit-for-bit
- the selector / objective mix does improve the frozen-`1201` base tradeoff:
  `12514` beats `12423` on base overall accuracy, final-query accuracy,
  route match, and exit timing
- but the held-confirm story did not change at all:
  `12514` and `12423` have the same `phase11_verify` metrics on
  `full_locked`, `finalquery_heavy`, and `longdistance`

So this family improved what the main split rewards, but it did not move the
actual held-confirm boundary.

## Conclusion

`gnn2-e9w` is closed as a rerun-backed positive for the base frozen-`1201`
tradeoff and a strong mapping result for held confirms.

The sharpened next question is no longer "try another simple selector mix".
It is whether content can be transferred more directly onto the improved
`12514` boundary. The next issue is `gnn2-28d`: source-logit content
distillation on the frozen-`1201` interpolation boundary.
