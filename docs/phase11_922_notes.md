# Phase 11 `gnn2-922` Notes

## Scope

`bxr` showed a narrow but stable pattern on frozen `1201`:

- lighter delayed-only `1874` source distillation improved base behavior
- the best `0.25` branch (`13011`) reran exactly
- but held-confirm metrics stayed fixed relative to the heavier delayed-only
  branches

This follow-up tested whether an even lighter delayed-only teacher signal could
recover the remaining base gap to `12713` while preserving that same held-confirm
plateau.

Nothing about the frozen `1201` core or the confirm-mix scaffold changed. The
only changes were:

- very-light delayed-only logits weight
- a delayed-start ablation for the light branch

## Variants

- `13111`: delayed-only `logits_weight=0.10`
- `13112`: delayed-only `logits_weight=0.15`
- `13113`: delayed-only `logits_weight=0.20`
- `13114`: delayed-only `logits_weight=0.15`, `start_step=48`
- `13111_rerun1`: exact rerun of the strongest scout

## Scout Results

Best observed validation checkpoints:

- `13111`: `0.6392 / 0.3555 / 0.7974 / 124.54`
- `13112`: `0.6470 / 0.3456 / 0.7756 / 124.07`
- `13113`: `0.6357 / 0.3515 / 0.7795 / 123.28`
- `13114`: `0.6348 / 0.3247 / 0.7944 / 124.63`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The useful split is:

- `13112` had the best base overall, but it gave back too much final-query
  accuracy and route fidelity
- `13111` was the best balanced point and the only branch that stayed close to
  `12713` on both base final-query accuracy and route
- `13113` and `13114` were clearly weaker nearby challengers and were stopped
  after the first validation interval

## Rerun And Confirm

`13111_rerun1` matched the original exactly through the decisive scout window:

- step `23`: `0.6392 / 0.3555 / 0.7974 / 124.54`
- step `47`: `0.6289 / 0.3376 / 0.7746 / 123.33`

`phase11_verify` on `13111`:

- `base`: `0.6309 / 0.3482 / 0.7918 / 124.73`
- `full_locked`: `0.1301 / 0.2530 / 0.9876 / 126.43`
- `finalquery_heavy`: `0.1990 / 0.2477 / 0.9860 / 126.44`
- `longdistance`: `0.1851 / 0.2642 / 0.9843 / 157.96`

For comparison:

- `13011` verify:
  `0.6287 / 0.3341 / 0.7835 / 124.23` on `base`,
  with the same held-confirm metrics
- `12713` verify:
  `0.6455 / 0.3623 / 0.7814 / 124.40` on `base`,
  `0.1224 / 0.2513 / 0.9866 / 126.20` on `full_locked`

## Interpretation

- pushing delayed-only teacher weight below `0.25` does not unlock a new regime
- `13111` is a real, rerun-backed tradeoff point:
  it slightly improves base route and exit relative to `12713`, and it improves
  base final-query accuracy relative to `13011`
- but it still does not close the frozen-`1201` base gap to `12713` on overall
  or final-query accuracy
- more importantly, the held-confirm story is unchanged:
  `13111`, `13011`, and the heavier delayed-only branches all sit on the same
  confirm plateau to displayed precision
- `13112` shows that optimizing base overall alone is not enough here:
  it improved overall at the cost of the more relevant final-query tradeoff
- `13113` and delayed-start `13114` add no evidence that the family has another
  nearby win

So the scientific conclusion is not "very-light delayed distillation fails
completely." It does move the base tradeoff surface. But the family still looks
stuck on a single held-confirm plateau, and even its best base-balanced point
does not beat the stronger frozen-`1201` incumbent.

## Conclusion

`gnn2-922` is closed as a rerun-backed mixed result.

The best branch, `13111`, is the cleanest delayed-only source-distill point so
far below `0.25`, but it still does not beat `12713` on the main frozen-`1201`
frontier and it does not move the held-confirm ceiling.

The next sharpened question is whether the stronger base behavior of `12713` and
the slightly better route-preserving delayed-distill point `13111` can be
combined directly instead of tuned only through scalar teacher weight. That
follow-up should test same-architecture checkpoint / head interpolation between
the two frozen-`1201` boundaries.
