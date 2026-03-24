# Phase 11 `gnn2-kc7` Notes

## Scope

Start from the same frozen `1874` multiview query-gated selector baseline used
through the rest of the phase-11 follow-ups. Keep routing, memory, control,
and the main answer head fixed, but insert a tiny trainable adapter only on the
multiview fusion path. The goal is to test whether the remaining held-confirm
gap comes from the fused read path being slightly too rigid, without reopening
the broader reader or touching upstream state.

## Variants

- `11711`: zero-init affine adapter on `multiview_fusion` only
- `11712`: rank-4 low-rank adapter on `multiview_fusion` only
- `11713`: affine fusion adapter + trainable `multiview_query_proj`
- `11714`: rank-4 fusion adapter + trainable `multiview_query_proj`

All four variants reused the same frozen phase-10 `1874` multiview
query-gated checkpoint and the same `full_locked` final-query-accuracy
selection rule used by `11101`.

## Validation Summary

- `11101` selector-only baseline: `overall 0.9937`, `fq_acc 0.9871`,
  `fq_route 0.9533`, `fq_exit 122.48`
- `11711`: `overall 0.2593`, `fq_acc 0.2810`, `fq_route 0.9474`,
  `fq_exit 121.61`
- `11712`: `overall 0.3193`, `fq_acc 0.3764`, `fq_route 0.9434`,
  `fq_exit 121.74`
- `11713`: `overall 0.3501`, `fq_acc 0.3198`, `fq_route 0.9295`,
  `fq_exit 120.68`
- `11714`: `overall 0.4971`, `fq_acc 0.5174`, `fq_route 0.9315`,
  `fq_exit 120.52`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The family-level pattern is unambiguous:

- even the best adapter variant stayed massively below the selector-only
  frozen-head baseline
- low-rank was safer than affine, but still nowhere near viable
- allowing `multiview_query_proj` to move improved answer fit relative to the
  pure fusion-only touches, but weakened route and still remained far below the
  incumbent
- the problem is not just that the frozen fusion path needs a tiny extra
  degree of freedom

## Reproducibility

- `11714_rerun1` matched `11714` exactly on validation summary metrics.

That is enough to retire the family as a reproducible negative rather than a
one-seed miss.

## Conclusion

Multiview-fusion-only adapters also fail on frozen `1874`. Relative to the
selector-only baseline:

- the smallest pure fusion touches collapse answer quality too severely
- adding `multiview_query_proj` freedom recovers part of that collapse, but not
  nearly enough
- route and exit remain relatively strong, which means the family is not
  breaking routing outright
- but the read-path touch is still too destructive to the strong frozen answer
  geometry to be useful

Taken together with the earlier phase-11 negatives, this further sharpens the
boundary: the remaining gap on frozen `1874` does not look like a problem that
can be solved by tiny answer-head swaps or by tiny fusion-path adapters alone.

## Recommended Next Step

Stop iterating on narrow frozen-`1874` read-path touches that merely perturb the
same decodable representation. The next narrow follow-up should test whether the
`11101`-style selector baseline reaches a meaningfully higher held-confirm
ceiling on the stronger `1201` upper-bound source, which would separate
source-quality limits from frozen-reader/objective limits more cleanly.
