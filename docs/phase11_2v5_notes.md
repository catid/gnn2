# Phase 11 `gnn2-2v5` Notes

## Scope

`m8a` exhausted scalar full-checkpoint interpolation between the frozen-`1201`
base boundary `12713` and the lighter held-confirm boundary `13111`:

- `13211` (`95/5`) was the best stable full-checkpoint blend
- `13312` (`96/4`) reproduced as a real transient, not a better stable point
- the next question was therefore narrower than another scalar blend sweep

This follow-up asked whether the transferable benefit from `13111` lives in a
small set of reader-side prefixes rather than in the whole checkpoint.

The frozen `1201` core stayed unchanged. The only thing varied here was which
prefixes were imported from `13111` into the stronger `12713` boundary before
training.

## Variants

- `13411`: import only `readout.` at `12713@0.95 + 13111@0.05`
- `13412`: import only `sink_proj` at `12713@0.95 + 13111@0.05`
- `13413`: import `sink_proj + readout.` at `12713@0.95 + 13111@0.05`
- `13415`: import only `readout.` at `12713@0.90 + 13111@0.10`
- `13411_rerun1`: exact rerun of the strongest prefix-only point

In practice the decisive movable prefixes in this source family were:

- `sink_proj.*`
- `readout.*`

No broader multiview-specific projection block existed here, so `2v5`
effectively became a direct test of `readout.` vs `sink_proj`.

## Scout Results

Best observed validation checkpoints:

- `13411`: `0.6523 / 0.3833 / 0.7766 / 124.08` at step `215`
- `13412`: `0.6675 / 0.3714 / 0.7994 / 124.69` at step `23`
- `13413`: `0.6455 / 0.3674 / 0.7766 / 123.86` at step `71`
- `13415`: `0.6265 / 0.3178 / 0.7925 / 124.45` at step `71`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The important split is:

- `13411` is the only branch that materially improved the frozen-`1201` base
  frontier over both `12713` and `13211`
- `13412` looked superficially viable at the first checkpoint, but it stalled
  immediately at step `23` and never justified promotion
- `13413` and `13415` both reached later checkpoints, but neither matched the
  `13411` base tradeoff

So the gain is not “reader-side prefixes” in general. It is specifically the
`readout.` prefix family.

## Confirm Comparison

`phase11_verify` on `13411`:

- `base`: `0.6479 / 0.3826 / 0.7884 / 123.97`
- `full_locked`: `0.1306 / 0.2539 / 0.9876 / 126.43`
- `finalquery_heavy`: `0.1980 / 0.2465 / 0.9860 / 126.44`
- `longdistance`: `0.1846 / 0.2635 / 0.9843 / 157.96`

For comparison:

- `12713 base`: `0.6436 / 0.3643 / 0.7894 / 123.93`
- `12713 full_locked`: `0.1224 / 0.2513 / 0.9866 / 126.20`
- `13211 base`: `0.6606 / 0.3635 / 0.7537 / 124.20`
- `13211 full_locked`: `0.1301 / 0.2530 / 0.9876 / 126.43`

This means:

- `13411` improves the main base tradeoff over the old stable checkpoint blend
  line
- it also slightly improves base final-query accuracy over `12713`
- and it preserves essentially the imported `13111/13211` held-confirm plateau

So `13411` is not a held-confirm breakthrough, but it is a real base-boundary
improvement with no confirm penalty.

## Rerun

`13411_rerun1` reproduced the full selected-checkpoint progression:

- original best moved through `23 -> 71 -> 143 -> 215`
- rerun matched the same progression and reached the same selected checkpoint
  at step `215`
- selected rerun validation metrics matched exactly:
  `0.6523 / 0.3833 / 0.7766 / 124.08`

That is strong evidence that the `13411` gain is not checkpoint noise or a
launch artifact. The readout-only prefix transfer is a real mechanism.

## Interpretation

- full-checkpoint interpolation was too blunt:
  it mixed useful and harmful pieces of `13111` together
- prefix-only transfer shows that the portable benefit lives in `readout.`
  rather than in `sink_proj`
- `sink_proj` alone is not the answer, and adding it back on top of `readout.`
  does not help
- the heavier `90/10` readout import already oversteps the local optimum

So the current map is sharper than before:

- the transferable improvement from `13111` is localized to readout geometry
- the best stable point is now `13411`, not `13211`
- the next step should stay narrow:
  start from the better `13411` readout-only boundary and test whether a
  confirm-aware selector or objective can move the held-confirm metrics
  without giving back the new base gain

## Conclusion

`gnn2-2v5` is closed as a rerun-backed positive.

The prefix-only experiment found a real new frozen-`1201` local optimum:
`13411` improves the base boundary over both `12713` and `13211`, while
preserving the imported held-confirm plateau. The scientific contribution is
also structural: the useful transfer from `13111` is specifically in
`readout.`, not in `sink_proj` or a broader full-checkpoint blend.
