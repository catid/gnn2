# Phase 11 `gnn2-28d` Notes

## Scope

`e9w` ended with a clear frozen-`1201` boundary:

- `12514` improved the base frozen-`1201` tradeoff over `12423`
- but `phase11_verify` showed that its held-confirm metrics were still identical
  to the earlier interpolation boundary

This follow-up kept the frozen `1201` core and the `12514` interpolation head
path fixed, and changed only the training signal by adding content-only source
logit distillation from the strong `1201` source.

## Variants

- `12611`: final-query-only source-logit distillation with `logits_weight=0.25`
- `12612`: final-query-only source-logit distillation with `logits_weight=0.5`
- `12613`: final-query-only source-logit distillation with `logits_weight=1.0`
- `12614`: all-scope source-logit distillation with `logits_weight=0.5`
- `12612_rerun1`: exact rerun of the least-bad scout

## Result

The whole family underperformed the plain `12514` boundary on the base tradeoff.

Selected base-test metrics:

- `12514`: `0.6423 / 0.3549 / 0.7721 / 124.62`
- `12611`: `0.6279 / 0.3255 / 0.7781 / 123.88`
- `12612`: `0.6273 / 0.3255 / 0.7828 / 124.28`
- `12613`: `0.5775 / 0.2934 / 0.7654 / 123.36`
- `12614`: `0.6354 / 0.3148 / 0.7801 / 123.66`
- `12612_rerun1`: exact match to `12612`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Confirm Comparison

`phase11_verify` on `12612`:

- `base`: `0.6175 / 0.3108 / 0.7921 / 124.21`
- `full_locked`: `0.1217 / 0.2500 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1937 / 0.2443 / 0.9885 / 126.36`
- `longdistance`: `0.1820 / 0.2571 / 0.9922 / 158.65`

For comparison `12514` verify remained:

- `base`: `0.6325 / 0.3483 / 0.7894 / 123.65`
- `full_locked`: `0.1217 / 0.2500 / 0.9866 / 126.20`
- `finalquery_heavy`: `0.1937 / 0.2443 / 0.9885 / 126.36`
- `longdistance`: `0.1820 / 0.2571 / 0.9922 / 158.65`

## Interpretation

- `12612` is not a one-seed miss:
  the exact rerun matched bit-for-bit
- source-logit distillation from the strong `1201` teacher did not improve the
  held-confirm ceiling at all:
  `12612` and `12514` have identical verify metrics on `full_locked`,
  `finalquery_heavy`, and `longdistance`
- the distillation family also weakened the base tradeoff:
  every scout lost final-query accuracy relative to `12514`
- pushing the teacher weight harder made the base tradeoff worse,
  and broadening the target scope from final-query-only to all cases did not
  recover the loss

So the family was not merely neutral on held confirms. It was strictly worse on
the main base tradeoff while leaving the confirm boundary unchanged.

## Conclusion

`gnn2-28d` is closed as a rerun-backed negative.

The sharpened next question is no longer "add teacher logits onto the `12514`
boundary". It is whether the `1201` frozen head path can be improved by
changing how the head is initialized or blended, rather than by adding another
teacher loss on top of the same objective.
