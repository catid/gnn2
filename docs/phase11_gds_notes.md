# Phase 11 `gnn2-gds` Notes

## Scope

Starting from `4nm`, soft route-aware composite selectors on the frozen `1201`
probe-initialized two-view reader failed in two distinct ways:

- most selectors collapsed into route-dead short-exit checkpoints
- the only late-exit selector `12113` still underperformed both `11914` and
  `12011`

This follow-up keeps the same frozen `1201` core and the same probe-initialized
reader, and changes only checkpoint selection:

- pure lexicographic ordering
- lexicographic ordering with a hard route floor

The question is whether route can be preserved if it is treated as a hard
ordering constraint rather than a soft weighted objective.

## Variants

- `12211`: lexicographic `full_locked fq_route -> overall`
- `12212`: lexicographic `full_locked fq_route -> fq_acc -> overall`
- `12213`: route-floor lexicographic `fq_route>=0.50 -> fq_route -> overall`
- `12214`: stricter route-floor lexicographic
  `fq_route>=0.55 -> fq_route -> fq_acc -> overall`
- `best_rerun`: exact rerun of the strongest scout, if competitive

## Result

The family is a rerun-backed negative.

Selected test metrics:

- `11914` probe-init boundary: `0.2598 / 0.3015 / 0.6317 / 107.91`
- `12011` selector-only boundary: `0.5264 / 0.3189 / 0.3964 / 122.12`
- `12113` soft composite boundary: `0.3346 / 0.2654 / 0.3803 / 107.06`
- `12211` lexicographic route -> overall:
  `0.2347 / 0.3115 / 0.0000 / 21.75`
- `12212` lexicographic route -> fq_acc -> overall:
  `0.3079 / 0.2841 / 0.1237 / 105.27`
- `12213` route-floor `0.50` -> route -> overall:
  `0.4323 / 0.2694 / 0.2647 / 84.75`
- `12213_rerun1`: exact match to `12213`
- `12214` route-floor `0.55` -> route -> fq_acc -> overall:
  `0.4323 / 0.2794 / 0.0922 / 59.93`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Interpretation

- pure lexicographic ordering is not enough:
  `12211` fell straight into a route-dead checkpoint and `12212` only recovered
  a weak late-exit regime with very low route match
- hard route floors also did not rescue the family:
  both floor variants selected checkpoints with a leading floor flag of `0.0`,
  meaning no validation checkpoint ever reached the required `full_locked`
  final-query route threshold
- `12213` is the least-bad member because it improves overall behavior relative
  to `11914` and `12113`, but it still gives back too much route and exits much
  earlier than `12011`
- `12214` shows that tightening the floor simply makes the selector chase the
  same route-poor regime without ever finding a route-satisfying checkpoint

## Conclusion

`gnn2-gds` is closed as a rerun-backed negative.

Hard route-floor and lexicographic checkpoint selection do not fix the frozen
`1201` probe-init tradeoff:

- the route-rich `11914` regime is not being missed by a soft selector
- the training trajectory itself does not present a checkpoint that satisfies a
  meaningful held-confirm route floor while keeping good overall behavior
- selector changes alone are now exhausted on this family

The next follow-up should move from selection-only changes to training-time
route-preserving regularization on the same frozen `1201` probe-init reader.
