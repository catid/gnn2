# Phase 11 `gnn2-j61` Notes

## Scope

`kwy` showed that nearby same-source head blending can improve the frozen-`1201`
base tradeoff, but does not move held confirms.

This follow-up asked a sharper question: instead of blending within the weak
`1201` family, can head geometry from the much stronger frozen-`1874` query-gated
reader transfer onto the frozen `1201` core?

To make that question fair, the family used four query-gated transfer variants:

- direct cross-source head transplant with native `1201` sink projection
- direct cross-source full-head transplant including sink projection
- softer `11011/11811` head blend `85/15`
- softer `11011/11811` head blend `70/30`

All four ran under the same `12514`-style confirm-mix / final-query-weighted
selector discipline.

## Variants

- `12811`: native `11811` sink projection + full `11011` query-gated head
- `12812`: full `11011` head including sink projection
- `12813`: native sink + `11011/11811` head blend `85/15`
- `12814`: native sink + `11011/11811` head blend `70/30`

## Result

This family is a clean negative.

Best observed validation points:

- `12811`: `0.4541 / 0.2463 / 0.0735 / 117.77`
- `12812`: `0.3647 / 0.2642 / 0.0606 / 118.41`
- `12813`: `0.3931 / 0.2562 / 0.0645 / 119.02`
- `12814`: `0.3984 / 0.2324 / 0.0556 / 118.31`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The two direct transfers were allowed to continue through the second validation
point and still remained near-route-dead. The softer blended variants were
stopped after the first validation interval because they showed the same failure
mode immediately.

## Interpretation

- direct `1874` head transfer onto frozen `1201` does not rescue route
- preserving the native `1201` sink projection does not solve that problem
- softer blends toward the native `11811` head do not help either
- the failure mode is the same as the earlier `11721/11723/11811/11814`
  portability attempts: reasonable train behavior, but final-query routing on
  validation remains near zero

So the new answer is sharper than "query-gated heads fail on `1201`".
Even importing strong-source query-gated geometry from a genuinely better frozen
source is not enough to make the `1201` portability problem route-faithful.

## Conclusion

`gnn2-j61` is closed as a four-variant negative.

The next question is no longer "can strong-source query-gated geometry be
ported onto frozen `1201`?" That answer is now no. The next issue is
`gnn2-b4t`: content distillation from the stronger frozen `1874` source onto the
best frozen-`1201` MLP boundary (`12713`), avoiding the failed query-gated
portability path entirely.
