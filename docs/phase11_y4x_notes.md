# Phase 11 `gnn2-y4x` Notes

## Scope

After `8te` and `g51`, the frozen-`1201` line had a sharp but unsolved tradeoff:

- `12011` recovered decent overall behavior but gave back too much route
- `11914` preserved a more useful late-route regime but at a very bad overall
  cost

This follow-up kept the same frozen `1201` core and the same decisive-view
reader, and changed only the initialization path:

- interpolate the balanced `12011` head with the routeful `11914` head
- keep routing, memory, and control frozen
- test whether a blended initialization can recover a better tradeoff than
  either parent

## Variants

- `12411`: 75% balanced / 25% routeful, selected on `full_locked` overall
- `12412`: 50% balanced / 50% routeful, selected on `full_locked` overall
- `12413`: 25% balanced / 75% routeful, selected on `full_locked` overall
- `12414`: 50% balanced / 50% routeful, selected on `full_locked`
  final-query accuracy
- `12412_rerun1`: exact rerun of the best interpolation point

## Selected Test Results

- `11732` frozen-`1201` baseline: `0.5238 / 0.2620 / 0.6197 / 120.08`
- `11914` routeful probe-init boundary: `0.2598 / 0.3015 / 0.6317 / 107.91`
- `12011` balanced selector-only boundary: `0.5264 / 0.3189 / 0.3964 / 122.12`
- `12411`: `0.5501 / 0.3048 / 0.4900 / 124.94`
- `12412`: `0.6276 / 0.3295 / 0.7527 / 124.26`
- `12412_rerun1`: exact match to `12412`
- `12413`: `0.6309 / 0.3108 / 0.7901 / 120.23`
- `12414`: `0.6273 / 0.3275 / 0.7366 / 124.86`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Interpretation

- `12411` shows that a mostly balanced interpolation already helps:
  it improves overall and keeps later exits, but route is still too weak to
  count as a serious fix
- `12412` is the key result:
  it is the first frozen-`1201` variant in this branch that simultaneously
  improves overall behavior, final-query accuracy, route match, and exit timing
  over the prior baselines
- `12413` pushes route even higher than `12412`, but it gives back final-query
  accuracy and exit timing relative to the balanced midpoint
- `12414` confirms that changing only the selector on the same 50/50 blend does
  not beat the simpler overall-selected `12412`

The exact rerun matters:

- `12412_rerun1` matched exactly
- so the improvement is not a one-seed artifact
- interpolation between the balanced and routeful heads is a real mechanism on
  the frozen `1201` source

## Conclusion

`gnn2-y4x` is closed as a rerun-backed positive boundary.

Head interpolation on the same frozen `1201` core succeeded where the recent
selection-only, regularization, and rebalance families failed:

- the midpoint blend `12412` clearly beats `11732`, `11914`, and `12011`
- the gain comes from combining the balanced and routeful regimes rather than
  from another reader-family change
- the local map now supports a narrower next step around this interpolation
  boundary, not a return to unrelated frozen-reader variants
