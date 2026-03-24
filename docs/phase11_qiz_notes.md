# Phase 11 `gnn2-qiz` Notes

## Scope

Port the best frozen `1874` selector-style readers onto the stronger frozen
`1201` ES upper-bound source. The point was to separate source-quality limits
from reader-portability limits: if the phase-11 selector baseline was being
held back mainly by source quality, then the same frozen reader recipe should
look better on `1201`.

## Variants

- `11721`: direct `11101`-style multiview selector port onto frozen `1201`
- `11722`: simpler readout-only selector port onto frozen `1201`
- `11723`: query-gated selector port onto frozen `1201`
- `11722_rerun1`: exact rerun of the least-bad readout-only port

All runs selected on `full_locked` final-query accuracy and started from the
phase-10 `1201` anchor checkpoint.

## Validation Summary

- `11721`: `overall 0.4810`, `fq_acc 0.2661`, `fq_route 0.0000`,
  `fq_exit 11.61`
- `11722`: `overall 0.4868`, `fq_acc 0.2522`, `fq_route 0.0000`,
  `fq_exit 29.75`
- `11723`: `overall 0.4771`, `fq_acc 0.2473`, `fq_route 0.0000`,
  `fq_exit 19.84`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Every direct frozen reader port collapsed into the immediate-exit,
route-dead basin. Stronger source quality by itself did not rescue the
phase-11 frozen reader recipe.

## Reproducibility

- `11722_rerun1` matched `11722` exactly.

That is enough to retire the direct frozen-port family as a reproducible
negative.

## Conclusion

The `1201` failure is not "the same reader, but still source-limited". The
family collapsed so completely that the more plausible diagnosis became an
interface mismatch between the `1201` source geometry and the frozen
`1874`-derived reader parameterization. That diagnosis motivated the follow-up
audit in `gnn2-t2d`.
