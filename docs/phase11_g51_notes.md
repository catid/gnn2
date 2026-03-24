# Phase 11 `gnn2-g51` Notes

## Scope

Start from the `1201` probe-init boundary after `8te` and `gds`:

- the frozen `1201` core plus decisive-view probe-init reader can recover a real
  late-exit regime
- but selector-only changes still fail to hold a good route/overall tradeoff
- hard route-aware selection also failed because no checkpoint ever reached a
  meaningful held-confirm route floor

This follow-up kept the same frozen `1201` core and the same probe-initialized
reader, but moved from checkpoint selection into training-time
route-preservation:

- use the strong `1201` anchor as a proxy teacher on confirm-like batches
- regularize route actions and/or wait-release behavior
- keep routing, memory, and control frozen

## Variants

- `12311`: route-action regularization on final-query-only cases
- `12312`: route-action regularization on delayed-only cases
- `12313`: wait/release regularization on final-query-only cases
- `12314`: combined route-action + wait/release regularization on
  final-query-only cases
- `12312_rerun1`: exact rerun of the least-bad delayed-only route-action
  variant

## Selected Test Results

- `11732` baseline: `0.5238 / 0.2620 / 0.6197 / 120.08`
- `11914` probe-init boundary: `0.2598 / 0.3015 / 0.6317 / 107.91`
- `12011` selector-only rebalance: `0.5264 / 0.3189 / 0.3964 / 122.12`
- `12213` route-floor selector: `0.4323 / 0.2694 / 0.2647 / 84.75`
- `12311`: `0.5091 / 0.3108 / 0.0000 / 21.26`
- `12312`: `0.5312 / 0.2988 / 0.0321 / 80.95`
- `12312_rerun1`: exact match to `12312`
- `12313`: `0.5684 / 0.2634 / 0.0180 / 54.45`
- `12314`: `0.4899 / 0.2594 / 0.0007 / 32.72`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Interpretation

- final-query-only route-action regularization `12311` is a clear wrong-basin
  failure:
  it keeps decent global fit but destroys held-confirm route and exits almost
  immediately
- delayed-only route-action regularization `12312` is the least-bad variant:
  it preserves late exits better than the other regularized runs, but route
  match stays near zero and far below even the weak selector-only boundaries
- wait/release-only regularization `12313` raises overall behavior but still
  fails to preserve the recovered route regime and exits much earlier than the
  useful frozen-reader baselines
- combined route-action + wait/release regularization `12314` is actively
  harmful:
  it collapses to a near-zero-route short-exit regime

The exact rerun of `12312` matters here:

- the least-bad training-time regularizer matched exactly
- so this is not a noisy one-off miss
- the whole family is a stable negative

## Conclusion

`gnn2-g51` is closed as a rerun-backed negative.

Training-time route-preserving regularization on the frozen `1201`
probe-initialized reader does not rescue the route/overall tradeoff:

- selection-only tweaks were already not enough
- adding light proxy-teacher route regularization during training is still not
  enough
- the least-bad delayed-only variant remains far below both the old useful
  route boundary (`11914`) and the better-balanced selector baseline (`12011`)

So the sharper next step is no longer “add another light route-regularizer”.
If the `1201` frozen-reader line continues, it should move to a different
mechanism than simple training-time agreement penalties.
