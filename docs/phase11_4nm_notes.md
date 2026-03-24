# Phase 11 `gnn2-4nm` Notes

## Scope

Starting from `8te`, the probe-initialized two-view frozen-`1201` reader had
two stable boundaries:

- `11914`: strong final-query route recovery but very poor overall behavior
- `12011`: much better overall behavior but too much route loss

This follow-up keeps the same frozen core and same probe-initialized reader, and
changes only the checkpoint-selection rule so route can explicitly influence the
best-checkpoint decision.

## Variants

- `12111`: weighted-sum selector on `full_locked` overall + final-query route
- `12112`: weighted-geomean selector on `full_locked` overall + route
- `12113`: weighted-geomean selector on `full_locked` overall + final-query
  accuracy + route
- `12114`: route-heavier weighted-geomean selector on `full_locked` overall +
  final-query accuracy + route
- `best_rerun`: exact rerun of the strongest scout, if competitive

## Result

All four route-aware composite selectors were negative on frozen `1201`.

- `12111` weighted-sum overall+route selected a route-dead checkpoint:
  `overall 0.4518 / fq_acc 0.2660 / fq_route 0.0000 / fq_exit 16.04`
- `12112` weighted-geomean overall+route also collapsed into a route-dead
  checkpoint:
  `overall 0.4893 / fq_acc 0.2888 / fq_route 0.0020 / fq_exit 26.54`
- `12113` weighted-geomean overall+fq_acc+route was the only selector that
  stayed in a late-exit regime but it still underperformed both incumbents:
  `overall 0.3346 / fq_acc 0.2654 / fq_route 0.3803 / fq_exit 107.06`
- `12114` route-heavier weighted-geomean reverted to the same short-exit
  route-dead behavior:
  `overall 0.4600 / fq_acc 0.2774 / fq_route 0.0000 / fq_exit 16.48`

Reference boundaries:

- `11914`: `overall 0.2598 / fq_acc 0.3015 / fq_route 0.6317 / fq_exit 107.91`
- `12011`: `overall 0.5264 / fq_acc 0.3189 / fq_route 0.3964 / fq_exit 122.12`

`12113_rerun1` matched `12113` exactly.

## Interpretation

Soft composite selection is not enough to preserve the recovered `11914`
final-query route regime while repairing global behavior. In practice the
selectors either:

- overweight overall behavior and fall into route-dead short-exit checkpoints
- or keep a late-exit regime while still losing both final-query route and
  final-query accuracy relative to the existing `11914` boundary

Adding final-query accuracy into the composite helped avoid the immediate-exit
collapse but still did not recover the stronger route match from `11914`, and
it remained well below the selector-only `12011` tradeoff on overall behavior.

## Conclusion

`gnn2-4nm` is a rerun-backed negative. Route-aware weighted-sum and
weighted-geomean checkpoint selection does not solve the frozen-`1201`
probe-init tradeoff. The next narrower follow-up should use a hard route floor
or lexicographic selector rather than another soft composite.
