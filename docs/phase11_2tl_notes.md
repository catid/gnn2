# Phase 11 `gnn2-2tl` Notes

## Scope

`2v5` localized the useful `13111 -> 1201` transfer to `readout.` prefixes and
produced a rerun-backed frozen-`1201` boundary at `13411`:

- `13411`: `0.6523 / 0.3833 / 0.7766 / 124.08`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

This follow-up kept the same frozen `1201` core and the same readout-only
`12713/13111` prefix blend, and changed only selector / objective details around
that `13411` boundary.

## Variants

- `13511`: proxy selector on `full_locked` final-query accuracy plus
  `final_query_weight=4.0`
- `13512`: direct selector on `selection_eval.full_locked.accuracy`
- `13513`: very light confirm-mix
  `full_locked + finalqueryheavy + longdistance` at `0.02` each, `fq=4`
- `13514`: same very light confirm-mix with `fq=5`
- `13511_rerun1`: exact rerun of the strongest scout

## Result

The family is a rerun-backed negative.

Selected validation metrics:

- `13411` readout-only boundary: `0.6523 / 0.3833 / 0.7766 / 124.08`
- `13511`: `0.6509 / 0.3635 / 0.7766 / 123.85`
- `13511_rerun1`: exact match to `13511`
- `13512`: `0.6484 / 0.3575 / 0.7726 / 124.38`
- `13513`: `0.6450 / 0.3525 / 0.7786 / 123.96`
- `13514`: `0.6436 / 0.3555 / 0.7835 / 124.56`

## Interpretation

- `13511` is the actual top challenger in this family, not the light
  confirm-mix variants
- the exact rerun matched bit-for-bit through the decisive checkpoints and
  selected the same best step `359`
- selector-only tuning can nearly recover the `13411` overall tradeoff, but it
  does not improve the deciding final-query accuracy
- direct `full_locked` overall selection (`13512`) is weaker than the proxy
  selector
- the very light confirm-mix variants (`13513`, `13514`) do not help:
  they trade small route / exit differences against weaker base overall accuracy
  and weaker final-query accuracy

So the family does not reveal a hidden better point around `13411`. It confirms
that narrow selector / light confirm-mix tuning is not enough to move the
readout-only prefix boundary.

## Conclusion

`gnn2-2tl` is closed as a rerun-backed negative.

The sharpened next question is no longer "try another tiny selector tweak" on
the same boundary. The more plausible next move is a direct content-transfer
experiment onto the stable `13411` readout-only prefix point, rather than more
lightweight checkpoint-selection or objective mixing around it.
