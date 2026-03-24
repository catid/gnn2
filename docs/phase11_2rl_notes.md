# Phase 11 `gnn2-2rl` Notes

## Scope

`2tl` showed that narrow selector and light confirm-mix changes around the stable
`13411` readout-only prefix boundary did not beat `13411` on the deciding base
tradeoff:

- `13411`: `0.6523 / 0.3833 / 0.7766 / 124.08`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

This follow-up kept the same frozen `1201` core and the same `12713/13111`
readout-only prefix blend as `13411`, and changed only one thing: delayed-only
source-logit content distillation from the stronger `13111` source.

## Variants

- `13611`: delayed-only source-logit distill, `lw=0.05`
- `13612`: delayed-only source-logit distill, `lw=0.10`
- `13613`: delayed-only source-logit distill, `lw=0.15`
- `13614`: `lw=0.10` with direct `selection_eval.overall` selection
- `13612_rerun1`: exact rerun of the strongest scout

## Result

The family is a rerun-backed mixed result, not a promoted improvement.

Selected validation metrics:

- `13411` readout-only boundary: `0.6523 / 0.3833 / 0.7766 / 124.08`
- `13611`: `0.6548 / 0.3734 / 0.7825 / 124.19`
- `13612`: `0.6650 / 0.3893 / 0.7855 / 124.19`
- `13613`: `0.6636 / 0.3843 / 0.7766 / 123.41`
- `13614`: `0.6587 / 0.3803 / 0.7696 / 123.14`
- `13612_rerun1`: exact match to `13612`

Confirm metrics for the winning branch:

- `13612` base: `0.6399 / 0.3598 / 0.7855 / 124.19`
- `13612` full_locked: `0.1306 / 0.2539 / 0.9876 / 126.43`
- `13612` finalquery_heavy: `0.1980 / 0.2465 / 0.9860 / 126.44`
- `13612` longdistance: `0.1846 / 0.2635 / 0.9843 / 157.96`

For comparison, `13411` had:

- base confirm: `0.6479 / 0.3826 / 0.7884 / 123.97`
- `full_locked`, `finalquery_heavy`, and `longdistance` confirms were
  effectively identical to `13612`

## Interpretation

- delayed-only `13111` logit distillation does create a real scout-time base
  gain on the stable `13411` boundary
- `lw=0.10` is the best local setting; `lw=0.15` gives back route and exit
  timing, while direct overall selection is weaker than the proxy selector
- the exact same-seed rerun matched bit-for-bit at the selected checkpoint
  (`step 335`), so the scout gain is real and reproducible
- however, phase-11 verify shows that the imported held-confirm plateau does not
  move at all relative to `13411`
- base confirm also slips slightly relative to `13411`, which means the scout
  gain does not survive the stronger confirmation protocol

So this is not a dead family in the sense of pure noise, but it is not a real
frontier improvement either. It reveals that delayed-only source-logit transfer
can improve the unconfirmed validation boundary while leaving the confirmed
behavior unchanged.

## Conclusion

`gnn2-2rl` is closed as a rerun-backed mixed result.

The next question, if this line is revisited, is no longer "does delayed-only
source distillation help at all?" That answer is yes on scout metrics. The
sharper question is whether the distillation schedule can be restricted or
reweighted so the scout gain survives confirm evaluation instead of washing out
to the same held-confirm plateau.
