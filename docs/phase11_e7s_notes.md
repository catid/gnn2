# Phase 11 `gnn2-e7s` Notes

## Scope

Start from the best frozen `1874` multiview query-gated reader and test confirmation-aware
objective mixing without touching memory, router, or control parameters.

## Variants

- `11011`: `full_locked@0.5`
- `11012`: `full_locked@0.75`, `final_query_weight=5.0`
- `11013`: `full_locked@0.5 + finalqueryheavy@0.5`, `final_query_weight=5.0`
- `11014`: `full_locked@0.4 + finalqueryheavy@0.3 + longdistance@0.3`, `final_query_weight=5.0`

All four variants reused the phase-10 `1874` frozen multiview query-gated baseline.

## Validation Summary

- `11011`: `overall 0.9907`, `fq_acc 0.9811`, `fq_route 0.9424`, `fq_exit 121.46`
- `11012`: `overall 0.9897`, `fq_acc 0.9791`, `fq_route 0.9404`, `fq_exit 121.14`
- `11013`: `overall 0.9922`, `fq_acc 0.9841`, `fq_route 0.9355`, `fq_exit 120.76`
- `11014`: `overall 0.9946`, `fq_acc 0.9891`, `fq_route 0.9414`, `fq_exit 121.61`

The mixed-shift variants improved base validation fit, but that did not transfer to held confirms.

## Confirm Results

Confirmed top variants:

- `11011`:
  - `base 0.9861 / 0.9724 / 0.9424 / 121.60`
  - `full_locked 0.6487 / 0.3144 / 0.8771 / 115.49`
  - `finalquery_heavy 0.4478 / 0.3116 / 0.8801 / 115.84`
  - `longdistance 0.5107 / 0.3015 / 0.8843 / 145.39`
- `11014`:
  - `base 0.9927 / 0.9860 / 0.9472 / 121.99`
  - `full_locked 0.6475 / 0.3121 / 0.8771 / 115.49`
  - `finalquery_heavy 0.4463 / 0.3098 / 0.8801 / 115.84`
  - `longdistance 0.5093 / 0.2994 / 0.8843 / 145.39`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Compared with the phase-9 strong-source baseline (`full_locked fq_acc 0.3237`, `fq_route 0.8771`,
`fq_exit 115.49`), the confirmation-aware mixes did not improve the held-confirm ceiling.

## Reproducibility

- `11011_rerun1` matched `11011` exactly on validation summary metrics.
- `11014_rerun1` matched `11014` exactly on validation summary metrics.

This makes the family a reproducible negative, not a one-seed miss.

## Conclusion

Confirmation-aware objective mixing changes base validation behavior but does not move the strong-source
held-confirm content ceiling on frozen `1874`. The remaining gap looks more like selection or objective
alignment than missing exposure to confirm-like distributions.

## Recommended Next Step

Test a narrower follow-up that keeps the same frozen `1874` baseline but changes how checkpoints are
selected or regularized against held-confirm proxies, rather than adding more data-mix variants.
