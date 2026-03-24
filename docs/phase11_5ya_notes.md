## gnn2-5ya

Question: after `gnn2-bkk` retired `final_query_only` timing changes, can the
useful `delayed_only` `13111` source-logit signal be shifted later and still
improve the stable `13711` frozen-`1201` boundary?

Baseline references:

- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`
- `13711` base confirm: `0.6392 / 0.3705 / 0.8024 / 124.42`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Late-start delayed-only sweep:

- `14211` `lw=0.10`, `start=24`, `stop=72`:
  `0.6689 / 0.3873 / 0.7944 / 124.31`
- `14212` `lw=0.10`, `start=48`, `stop=96`:
  `0.6450 / 0.3714 / 0.8004 / 124.04`
- `14213` `lw=0.15`, `start=24`, `stop=72`:
  `0.6699 / 0.3942 / 0.7954 / 124.21`
- `14214` `lw=0.15`, `start=48`, `stop=96`:
  `0.6743 / 0.3972 / 0.7627 / 123.95`

Proxy `full_locked` view at the selected checkpoints:

- `14211`: `0.6406 / 0.3735 / 0.7918 / 124.26`
- `14212`: `0.6328 / 0.3754 / 0.8067 / 124.48`
- `14213`: `0.6499 / 0.4034 / 0.7638 / 124.21`
- `14214`: `0.6450 / 0.3810 / 0.7890 / 123.36`

Only `14213` earned rerun.

Rerun:

- `14213_rerun1` matched exactly:
  `0.6699 / 0.3942 / 0.7954 / 124.21`

Confirm:

- `14213` base confirm: `0.6460 / 0.3700 / 0.7782 / 123.73`
- `14213` full_locked: `0.1306 / 0.2539 / 0.9876 / 126.43`
- `14213` finalquery_heavy: `0.1980 / 0.2465 / 0.9860 / 126.44`
- `14213` longdistance: `0.1846 / 0.2635 / 0.9843 / 157.96`

Interpretation:

- `14211` shows that starting the delayed-only signal at `24` can raise overall
  fit, but it does not quite preserve the `13711` final-query balance
- `14212` keeps route closest to `13711`, but it gives back too much
  final-query content to count as a better boundary
- `14213` is a real, reproducible schedule effect: it improved selected
  validation `overall` and `fq_acc` over `13711` while only slightly lowering
  selected-route fidelity
- `14214` pushes that tradeoff too far: it raises unconfirmed content further,
  but route drops too much to stay competitive
- the confirm pass is decisive: `14213` falls back onto the same held-confirm
  plateau and its base confirm route is worse than `13711`

Conclusion:

- late-start delayed-only timing is not noise; it genuinely reshapes the dev
  selector surface
- but it still does not produce a confirmed improvement over `13711`
- so `gnn2-5ya` closes as a rerun-backed mixed result rather than a promotion
