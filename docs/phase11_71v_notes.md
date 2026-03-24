# Phase 11 `gnn2-71v` Notes

## Scope

Start from the frozen `1874` multiview query-gated baseline and test whether
checkpoint selection against confirm-like proxy evaluations can move the held-confirm
ceiling without changing the reader, routing, memory, or training data mix.

## Variants

- `11101`: select on `full_locked` final-query accuracy
- `11102`: select on `full_locked` overall accuracy
- `11103`: select on `finalqueryheavy` final-query accuracy
- `11104`: select on `longdistance` final-query accuracy

All four variants reused the same frozen phase-10 `1874` baseline and changed only
the validation-time proxy used for checkpoint selection.

## Validation Summary

- `11101`: `overall 0.9937`, `fq_acc 0.9871`, `fq_route 0.9533`, `fq_exit 122.48`
- `11102`: `overall 0.9863`, `fq_acc 0.9722`, `fq_route 0.9414`, `fq_exit 121.80`
- `11103`: `overall 0.9878`, `fq_acc 0.9752`, `fq_route 0.9374`, `fq_exit 121.00`
- `11104`: `overall 0.9849`, `fq_acc 0.9692`, `fq_route 0.9355`, `fq_exit 121.10`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The direct split between selectors is clean:

- selecting on `full_locked` final-query accuracy (`11101`) was the strongest validation-time proxy
- selecting on `full_locked` overall accuracy (`11102`) was clearly weaker
- shifted proxies (`11103`, `11104`) fit their own proxy benchmarks but did not improve main validation

## Confirm Result For Lead Selector

Confirmed `11101`:

- `base 0.9890 / 0.9782 / 0.9550 / 122.65`
- `full_locked 0.6475 / 0.3121 / 0.8771 / 115.49`
- `finalquery_heavy 0.4463 / 0.3098 / 0.8801 / 115.84`
- `longdistance 0.5093 / 0.2994 / 0.8843 / 145.39`

Compared with the phase-9 strong-source baseline, the held-confirm ceiling did not move:

- phase-9 strong-source baseline: `full_locked fq_acc 0.3237`, `fq_route 0.8771`, `fq_exit 115.49`
- `11101`: `full_locked fq_acc 0.3121`, `fq_route 0.8771`, `fq_exit 115.49`

## Reproducibility

- `11101_rerun1` matched `11101` exactly on validation summary metrics.

That makes proxy-selection a reproducible negative for this source family, not a one-seed miss.

## Conclusion

Changing checkpoint selection to confirm-like proxy metrics alters which validation checkpoint is chosen,
but it does not improve the frozen `1874` held-confirm content ceiling. The remaining limit looks more like
the objective shape or the training trajectory itself than a bad early-stopping signal.

## Recommended Next Step

Keep the frozen `1874` baseline and move one notch narrower: test an explicit held-confirm proxy regularizer
or consistency penalty during training, rather than more proxy-selection variants.
