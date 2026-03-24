# Phase 11 `gnn2-87y` Notes

## Scope

`922` left a sharper frozen-`1201` picture:

- `12713` was still the best base boundary
- `13111` imported the slightly better delayed-distill held-confirm plateau
- but `13111` still trailed `12713` on base overall and base final-query
  accuracy

This follow-up tested whether those two nearby boundaries could be combined
directly through same-architecture checkpoint / head interpolation, without
touching the frozen `1201` core or changing the training recipe.

The only thing varied here was the blend weight between:

- the stronger base boundary `12713`
- the slightly better route/held-confirm boundary `13111`

## Variants

- `13211`: `12713@0.95 + 13111@0.05`
- `13212`: `12713@0.90 + 13111@0.10`
- `13213`: `12713@0.85 + 13111@0.15`
- `13211_rerun1`: exact rerun of the strongest scout

## Scout Results

Best observed validation checkpoints:

- `13211`: `0.6606 / 0.3635 / 0.7537 / 124.20`
- `13212`: `0.6304 / 0.3396 / 0.7716 / 124.12`
- `13213`: `0.6289 / 0.3297 / 0.7716 / 124.53`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

The first look is deceptive here:

- `13211` immediately improved base overall and essentially matched the old
  base final-query accuracy
- but its route dropped too much at the first checkpoint, so it had to be kept
  alive to the proxy-selected window
- `13212` and `13213` were already weaker nearby points and were retired early

Relevant later scout checkpoints for `13211`:

- `val` step `47`: `0.6519 / 0.3575 / 0.7726 / 122.88`
- `val_proxy_full_locked` step `47`: `0.6338 / 0.3604 / 0.7862 / 124.41`
- `val_proxy_full_locked` step `95`: `0.6274 / 0.3492 / 0.7993 / 124.55`

So the useful lesson is that the interpolation *does* import the `13111`
confirm-like plateau, but it still rides a base tradeoff rather than becoming a
clear dominance point over `12713`.

## Rerun And Confirm

`13211_rerun1` matched the original exactly through the decisive scout window:

- step `23`: `0.6606 / 0.3635 / 0.7537 / 124.20`
- step `47`: `0.6519 / 0.3575 / 0.7726 / 122.88`

`phase11_verify` on `13211`:

- `base`: `0.6423 / 0.3511 / 0.7855 / 124.12`
- `full_locked`: `0.1301 / 0.2530 / 0.9876 / 126.43`
- `finalquery_heavy`: `0.1987 / 0.2474 / 0.9860 / 126.44`
- `longdistance`: `0.1848 / 0.2639 / 0.9843 / 157.96`

For comparison:

- `12713` verify:
  `0.6455 / 0.3623 / 0.7814 / 124.40` on `base`,
  `0.1224 / 0.2513 / 0.9866 / 126.20` on `full_locked`
- `13111` verify:
  `0.6309 / 0.3482 / 0.7918 / 124.73` on `base`,
  `0.1301 / 0.2530 / 0.9876 / 126.43` on `full_locked`

## Interpretation

- interpolation between `12713` and `13111` is a real mechanism:
  the exact rerun matched through the full scout window
- the small `95/5` blend successfully transferred the `13111` held-confirm
  plateau:
  `13211` has the same confirm metrics as `13111`
- but `13211` still does not beat `12713` on the main base frontier:
  it gives back a little overall accuracy and final-query accuracy
- what it does improve is the tradeoff shape:
  it lands between the two parents, keeping slightly better base route than
  `12713` while staying much closer to `12713` than `13111` does
- the heavier nearby blends were not the answer:
  `13212` and `13213` were already weaker at the first scout interval

So this family is not a dead negative. It shows that the better held-confirm
plateau from the delayed-distill branch can be imported into a more
base-preserving checkpoint. But the 95/5 point is still not the final answer.

## Conclusion

`gnn2-87y` is closed as a rerun-backed mixed positive.

Checkpoint interpolation between `12713` and `13111` works and transfers the
`13111` confirm plateau into a checkpoint much closer to the `12713` base
regime. The best point tested, `13211`, still does not beat `12713` on base
overall or base final-query accuracy, so it is not a clean frontier move.

The next sharpened question is whether a *finer* interpolation near `12713`
(`97.5/2.5`, `96/4`, `94/6`, etc.) can recover the remaining base gap while
keeping the imported held-confirm plateau.
