# Phase 11 `gnn2-bxr` Notes

## Scope

`8bn` showed a real but mixed effect from strong-source logits on frozen `1201`:

- final-query-only distillation was noncompetitive
- delayed-only distillation (`12914`) reran exactly
- `phase11_verify` showed slightly better held-confirm metrics than `12713`
- but base overall and base final-query accuracy got worse than `12713`

This follow-up kept the same frozen `1201` core and the same delayed-only
`9142` teacher scope, and changed only:

- the delayed-only logits weight
- the selector metric

The goal was to see whether the small held-confirm gain from `12914` could be
kept while recovering the lost base tradeoff.

## Variants

- `13011`: delayed-only `logits_weight=0.25`, proxy `full_locked` final-query selector
- `13012`: delayed-only `logits_weight=0.35`, proxy `full_locked` final-query selector
- `13013`: delayed-only `logits_weight=0.25`, proxy `full_locked` overall selector
- `13014`: delayed-only `logits_weight=0.35`, proxy `full_locked` overall selector
- `13011_rerun1`: exact rerun of the strongest scout

## Scout Results

Best observed validation checkpoints:

- `13011`: `0.6514 / 0.3585 / 0.7934 / 124.02`
- `13012`: `0.6411 / 0.3357 / 0.7795 / 123.98`
- `13013`: `0.6294 / 0.3357 / 0.7855 / 124.48`
- `13014`: `0.6348 / 0.3267 / 0.7835 / 124.37`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

`13011` is the only branch that improved the `12914` base tradeoff. The other
three variants were cleanly weaker and were stopped once that was clear.

## Rerun And Confirm

`13011_rerun1` matched the original exactly through the full scout window that
decided promotion:

- step `23`: `0.6514 / 0.3585 / 0.7934 / 124.02`
- step `47`: `0.6426 / 0.3327 / 0.7766 / 124.29`

`phase11_verify` on `13011`:

- `base`: `0.6287 / 0.3341 / 0.7835 / 124.23`
- `full_locked`: `0.1301 / 0.2530 / 0.9876 / 126.43`
- `finalquery_heavy`: `0.1990 / 0.2477 / 0.9860 / 126.44`
- `longdistance`: `0.1851 / 0.2642 / 0.9843 / 157.96`

For comparison:

- `12914` verify:
  `0.6204 / 0.3245 / 0.7763 / 123.76` on `base`,
  with the same held-confirm metrics
- `12713` verify:
  `0.6455 / 0.3623 / 0.7814 / 124.40` on `base`,
  `0.1224 / 0.2513 / 0.9866 / 126.20` on `full_locked`

## Interpretation

- lowering delayed-only distill weight from `0.50` to `0.25` does help the base
  tradeoff:
  `13011` clearly beats `12914` on base overall and base final-query accuracy
- but the held-confirm metrics stayed effectively fixed:
  `13011` and `12914` produced the same confirm numbers to the displayed
  precision
- the overall-selector half was not the answer:
  both `13013` and `13014` were already weaker than `13011` at the first
  validation checkpoint
- even the improved `13011` tradeoff still does not beat `12713` on the main
  frozen-`1201` base frontier

So the actual lesson is narrower than "delayed-only distillation fails":
it does produce a stable tradeoff surface, but that surface seems to share a
single held-confirm plateau while the base behavior moves with weight.

## Conclusion

`gnn2-bxr` is closed as a rerun-backed mixed result.

The best branch, `13011`, kept the small held-confirm gain seen in `12914` while
recovering some of the lost base behavior, but it still did not beat the better
frozen-`1201` base boundary `12713`.

The next sharpened question is whether very-light delayed-only distillation can
recover the remaining base gap while preserving that same held-confirm plateau.
That follow-up is `gnn2-r1x`.
