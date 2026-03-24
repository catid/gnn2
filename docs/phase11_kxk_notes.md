# Phase 11 `gnn2-kxk` Notes

## Scope

Start from the same frozen `1874` multiview query-gated selector baseline used
through the rest of the phase-11 follow-ups. Keep routing, memory, control,
and the frozen multiview reader path fixed, but replace the single answer head
with a tiny mixture-of-heads reader. The goal is to test whether the remaining
held-confirm gap comes from needing multiple specialized answer subspaces
rather than one global head.

## Variants

- `11611`: 2 linear heads, mixture gate from the frozen readout input
- `11612`: 4 linear heads, mixture gate from the frozen readout input
- `11613`: 2 hidden heads (`h=64`), query-conditioned mixture gate
- `11614`: 4 hidden heads (`h=64`), query-conditioned mixture gate, light
  balance regularization (`0.01`)

All four variants reused the same frozen phase-10 `1874` multiview
query-gated checkpoint and the same `full_locked` final-query-accuracy
selection rule introduced in `11101`.

## Validation Summary

- `11101` selector-only baseline: `overall 0.9937`, `fq_acc 0.9871`,
  `fq_route 0.9533`, `fq_exit 122.48`
- `11611`: `overall 0.9844`, `fq_acc 0.9682`, `fq_route 0.9364`,
  `fq_exit 121.28`
- `11612`: `overall 0.9868`, `fq_acc 0.9732`, `fq_route 0.9414`,
  `fq_exit 121.37`
- `11613`: `overall 0.9785`, `fq_acc 0.9563`, `fq_route 0.9414`,
  `fq_exit 121.24`
- `11614`: `overall 0.9824`, `fq_acc 0.9643`, `fq_route 0.9355`,
  `fq_exit 120.86`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The family pattern is again consistent:

- all four mixtures stayed clearly below `11101`
- extra branches did not recover the route gap
- query-conditioned gating did not help
- light balance regularization did not help

## Mixture Diagnostics

- `11611`: entropy `0.315`, top-1 gate weight `0.879`
- `11612`: entropy `0.545`, top-1 gate weight `0.822`
- `11613`: entropy `0.229`, top-1 gate weight `0.930`
- `11614`: entropy `0.292`, top-1 gate weight `0.929`, balance loss `0.0865`

So the family did use multiple branches to some degree, especially the plain
4-head version, but that extra specialization still did not convert into better
route-preserving answer quality.

## Reproducibility

- `11612_rerun1` matched `11612` exactly on validation summary metrics.

That is enough to retire the family as a reproducible negative rather than a
one-seed miss.

## Conclusion

Mixture-of-heads readers also fail on frozen `1874`. Relative to the
selector-only baseline:

- simple 2-head and 4-head mixtures both weaken route and exit
- query-conditioned gating makes the heads more specialized, but not better
- adding branch capacity does not fix the ceiling
- balance regularization changes branch usage without improving the final
  tradeoff

Taken together with the earlier phase-11 negatives, this further sharpens the
boundary: the remaining gap does not look like a single-head geometry problem,
and it also does not look like a simple missing-specialization problem inside a
frozen answer head.

## Recommended Next Step

Stop iterating on purely frozen-head architectural swaps. The next narrow
follow-up should move one step closer to the read path itself while still
preserving routing: a tiny **multiview fusion-only adapter** on frozen `1874`,
with the answer head kept simple and the selection rule still tied to
`full_locked` final-query accuracy.
