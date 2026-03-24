# Phase 11 `gnn2-m8a` Notes

## Scope

`87y` established a useful but incomplete frozen-`1201` result:

- `13211` (`95/5`) was the best `12713/13111` checkpoint blend
- it imported the `13111` held-confirm plateau
- but it still trailed `12713` slightly on base overall and base final-query
  accuracy

This follow-up tested whether a *finer* interpolation nearer `12713` could
recover that remaining base gap without giving back the imported plateau.

Nothing else about the frozen `1201` recipe changed. The only thing varied here
was the checkpoint blend weight between the stronger base boundary `12713` and
the slightly better held-confirm boundary `13111`.

## Variants

- `13311`: `12713@0.975 + 13111@0.025`
- `13312`: `12713@0.96 + 13111@0.04`
- `13313`: `12713@0.94 + 13111@0.06`
- `13314`: `12713@0.925 + 13111@0.075`
- `13312_rerun1`: exact rerun of the strongest new scout

## Scout Results

Best observed validation checkpoints:

- `13311`: `0.6323 / 0.3347 / 0.7974 / 124.16`
- `13312`: `0.6455 / 0.3674 / 0.7875 / 124.15`
- `13313`: `0.6440 / 0.3396 / 0.7825 / 123.62`
- `13314`: `0.6504 / 0.3396 / 0.7795 / 124.21`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The important split is:

- `13311` was too close to `12713`: it kept decent route but lost too much
  base overall and final-query accuracy to justify the smaller `13111`
  injection
- `13312` looked like the only real challenger at the first checkpoint:
  it nearly matched `12713` on base overall while improving route relative to
  `13211`
- `13313` and `13314` never improved on the old `95/5` blend and were clearly
  weaker nearby points

The problem with `13312` is that the gain was not durable. Relevant later
checkpoints:

- `val` step `47`: `0.6279 / 0.3446 / 0.7776 / 123.76`
- `val` step `71`: `0.6357 / 0.3446 / 0.7676 / 124.07`
- `val_proxy_full_locked` step `23`: `0.6309 / 0.3688 / 0.7768 / 123.15`
- `val_proxy_full_locked` step `47`: `0.6177 / 0.3436 / 0.7759 / 123.35`

So the closer `96/4` blend produced a real transient, but it decayed back below
the earlier `95/5` local optimum instead of stabilizing into a better point.

## Rerun

`13312_rerun1` matched the original exactly through the decisive scout window:

- step `23`: `0.6455 / 0.3674 / 0.7875 / 124.15`
- step `47`: `0.6279 / 0.3446 / 0.7776 / 123.76`
- step `71`: `0.6357 / 0.3446 / 0.7676 / 124.07`

That matters because it shows the `96/4` gain was not noise or launch
instability. It is a reproducible transient that decays the same way on rerun.

## Interpretation

- fine interpolation near `12713` does *not* reveal a cleaner point than
  `13211`
- `13312` is scientifically useful because it shows the local geometry more
  sharply:
  moving from `95/5` to `96/4` briefly recovers the base gap, but the better
  tradeoff is not stable under the same training recipe
- `13311` shows that moving even closer to `12713` simply loses the imported
  `13111` benefit
- `13313` and `13314` show that pushing heavier than `96/4` also does not help

So the current picture is:

- the earlier `95/5` blend remains the stable local optimum
- nearby scalar checkpoint weights are exhausted
- the remaining promising direction is *narrower than full-checkpoint
  interpolation*, not another scalar blend sweep

## Conclusion

`gnn2-m8a` is closed as a rerun-backed negative around the existing `95/5`
boundary.

The best new candidate, `13312`, reran exactly but decayed the same way, so the
extra base gain was a real transient rather than a durable frontier move.
`13211` remains the best stable `12713/13111` checkpoint interpolation point.

The next sharpened question is whether the imported `13111` benefit can be
transferred *only through the decisive head/readout prefixes* rather than
through a full-checkpoint blend.
