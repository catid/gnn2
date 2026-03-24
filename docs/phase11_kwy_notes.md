# Phase 11 `gnn2-kwy` Notes

## Scope

`28d` showed that source-logit distillation did not improve the frozen-`1201`
`12514` boundary:

- `12612` reran exactly
- `phase11_verify` stayed identical to `12514` on
  `full_locked`, `finalquery_heavy`, and `longdistance`
- the whole family weakened the base tradeoff

This follow-up kept the frozen `1201` core and the `12514`
selector / objective fixed, and changed only the head-path geometry by blending
the strongest nearby route-heavier boundaries back into `12514`.

## Variants

- `12711`: `12514` / `11914` head blend `85 / 15`
- `12712`: `12514` / `11914` head blend `70 / 30`
- `12713`: `12514` / `12424` head blend `85 / 15`
- `12714`: `12514` / `12424` head blend `70 / 30`
- `12713_rerun1`: exact rerun of the strongest scout

## Result

The `11914` blends over-corrected toward route, while the light `12424` blend
produced a real base-tradeoff improvement.

Selected base-test metrics:

- `12514`: `0.6423 / 0.3549 / 0.7721 / 124.62`
- `12711`: `0.6315 / 0.3209 / 0.8222 / 123.63`
- `12712`: `0.6380 / 0.3222 / 0.8155 / 122.06`
- `12713`: `0.6436 / 0.3643 / 0.7894 / 123.93`
- `12714`: `0.6302 / 0.3175 / 0.7794 / 124.00`
- `12713_rerun1`: exact match to `12713`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Confirm Comparison

`phase11_verify` on `12713`:

- `base`: `0.6455 / 0.3623 / 0.7814 / 124.40`
- `full_locked`: `0.1224 / 0.2513 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1940 / 0.2447 / 0.9885 / 126.36`
- `longdistance`: `0.1833 / 0.2590 / 0.9922 / 158.65`

For comparison `12514` verify remained:

- `base`: `0.6325 / 0.3483 / 0.7894 / 123.65`
- `full_locked`: `0.1217 / 0.2500 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1937 / 0.2443 / 0.9885 / 126.36`
- `longdistance`: `0.1820 / 0.2571 / 0.9922 / 158.65`

## Interpretation

- `12713` is not a one-seed artifact:
  the exact rerun matched bit-for-bit
- blending a small amount of the route-heavier `12424` head into `12514`
  improves the base frozen-`1201` tradeoff:
  `12713` beats `12514` on base overall accuracy, final-query accuracy,
  and final-query route match
- the same trick does not move the held-confirm frontier in a meaningful way:
  the confirm ladders are effectively unchanged, with only tiny
  measurement-scale gains in `overall` and `fq_acc`

So head blending can improve the selected base boundary, but it still does not
solve the real held-confirm limit.

## Conclusion

`gnn2-kwy` is closed as a rerun-backed positive for the base frozen-`1201`
tradeoff and a strong mapping result for held confirms.

The sharpened next question is no longer whether a nearby same-source head blend
helps. It is whether held-confirm content can be transferred from a stronger
decodable source onto the improved frozen-`1201` boundary. The next issue is
`gnn2-j61`: cross-source head interpolation from strong frozen `1874` into
the frozen `1201` boundary.
