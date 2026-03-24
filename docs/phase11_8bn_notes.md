# Phase 11 `gnn2-8bn` Notes

## Scope

`j61` showed that direct `1874 -> 1201` query-gated head transfer was too abrupt:
all four cross-source query-gated variants stayed near-route-dead on the frozen
`1201` target.

This follow-up kept the better frozen-`1201` MLP boundary `12713` and tested a
much narrower mechanism:

- keep the frozen `1201` core and native `12713` head geometry
- add `9142` source-logit distillation only
- vary the distillation scope and schedule instead of transplanting head weights

## Variants

- `12911`: `9142` logits distillation on final-query cases only, `logits_weight=0.25`
- `12912`: same as `12911` with `logits_weight=0.50`
- `12913`: final-query-only `logits_weight=0.50` starting late at step `48`
- `12914`: delayed-only `logits_weight=0.50`
- `12914_rerun1`: exact same-seed rerun of the strongest scout

## Scout Results

Best selected validation checkpoints:

- `12911`: `0.6392 / 0.3317 / 0.7885 / 124.14`
- `12912`: `0.6406 / 0.3386 / 0.7686 / 123.69`
- `12913`: `0.6348 / 0.3337 / 0.7885 / 124.70`
- `12914`: `0.6504 / 0.3565 / 0.7944 / 123.97`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Relative to the frozen-`1201` incumbent `12713`
(`0.6436 / 0.3643 / 0.7894 / 123.93`), the family answer is:

- final-query-only distillation did not help
- late-start final-query-only distillation also did not help
- delayed-only distillation was the only competitive branch

## Rerun And Confirm

`12914_rerun1` matched `12914` exactly through the full scout window:

- step `23`: `0.6504 / 0.3565 / 0.7944 / 123.97`
- step `47`: `0.6499 / 0.3565 / 0.7795 / 124.21`
- step `71`: `0.6362 / 0.3267 / 0.7865 / 124.26`

`phase11_verify` on `12914`:

- `base`: `0.6204 / 0.3245 / 0.7763 / 123.76`
- `full_locked`: `0.1301 / 0.2530 / 0.9876 / 126.43`
- `finalquery_heavy`: `0.1990 / 0.2477 / 0.9860 / 126.44`
- `longdistance`: `0.1851 / 0.2642 / 0.9843 / 157.96`

For comparison, `12713` verify remained:

- `base`: `0.6455 / 0.3623 / 0.7814 / 124.40`
- `full_locked`: `0.1224 / 0.2513 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1940 / 0.2447 / 0.9885 / 126.36`
- `longdistance`: `0.1833 / 0.2590 / 0.9922 / 158.65`

## Interpretation

- `12914` is not a one-seed artifact:
  the exact rerun matched bit-for-bit through the scout window
- strong-source logits can influence the frozen-`1201` tradeoff without
  destroying route
- but the effect is mixed rather than clearly positive:
  `12914` gives up base overall and base final-query accuracy versus `12713`
  while only slightly improving held-confirm accuracy and route metrics
- this means source-logit distillation is not a clean fix on its own, but it is
  a real axis in the tradeoff map

## Conclusion

`gnn2-8bn` is closed as a rerun-backed mixed result:
delayed-only `1874` logits distillation shifts frozen-`1201` behavior slightly
toward held confirms, but it does not beat the better frozen-`1201` base
boundary `12713`.

The sharpened next question is no longer "does source-logit distillation do
anything at all?". It is whether the delayed-only distillation tradeoff can be
rebalanced to recover the lost base behavior while keeping the small held-
confirm gain. The next issue is `gnn2-4nm`.
