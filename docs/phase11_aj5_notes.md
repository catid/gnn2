# Phase 11 `gnn2-aj5` Notes

## Scope

`y4x` established a real interpolation improvement on frozen `1201`:

- `12412` beat the earlier `11732`, `11914`, and `12011` boundaries together
- but it was still unclear whether 50/50 was a true local optimum or just the
  first useful point

This follow-up stayed on exactly the same frozen `1201` core and exactly the
same decisive-view reader, and changed only the interpolation weight around the
new midpoint boundary.

## Variants

- `12421`: 60% balanced / 40% routeful
- `12422`: 55% balanced / 45% routeful
- `12423`: 45% balanced / 55% routeful
- `12424`: 40% balanced / 60% routeful
- `12423_rerun1`: exact rerun of the strongest nearby point

All variants kept:

- frozen routing, memory, and control
- the same `full_locked` overall selector
- the same final-query-weighted objective

## Selected Test Results

- `12412` previous boundary: `0.6276 / 0.3295 / 0.7527 / 124.26`
- `12421`: `0.6169 / 0.3202 / 0.6598 / 125.19`
- `12422`: `0.6126 / 0.3015 / 0.6905 / 125.70`
- `12423`: `0.6344 / 0.3215 / 0.7674 / 123.75`
- `12423_rerun1`: exact match to `12423`
- `12424`: `0.6120 / 0.2934 / 0.7948 / 123.35`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Interpretation

- moving only slightly toward the balanced side (`12421`, `12422`) gives back
  too much route without enough compensation elsewhere
- moving too far toward the routeful side (`12424`) raises route further, but it
  gives back both overall behavior and final-query accuracy
- the local sweet spot is `12423`:
  it improves on `12412` in overall behavior and route while giving back only a
  small amount of final-query accuracy and exit time

The rerun matters:

- `12423_rerun1` matched exactly
- so the local optimum shift is stable, not a noisy scout
- the interpolation family now has a reproducible best point at 45/55

## Conclusion

`gnn2-aj5` is closed as a rerun-backed positive refinement.

Local tuning around the interpolation boundary succeeded:

- `12412` was not yet the final optimum
- a slightly route-heavier blend (`12423`) is better overall for this frozen
  `1201` line
- the next disciplined step is no longer "keep sweeping weights blindly", but to
  promote and confirm the new `12423` optimum
