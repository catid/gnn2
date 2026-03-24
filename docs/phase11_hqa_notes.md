## gnn2-hqa

Question: after `5ya` showed a real late-start delayed-only effect at
`start=24 stop=72`, is there a narrow intermediate `logits_weight` between
`0.10` and `0.15` that keeps the selected-content gain from `14213` while
recovering enough route to improve the confirmed tradeoff over `13711`?

Baseline references:

- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`
- `13711` base confirm: `0.6392 / 0.3705 / 0.8024 / 124.42`
- `14213` selected val: `0.6699 / 0.3942 / 0.7954 / 124.21`
- `14213` base confirm: `0.6460 / 0.3700 / 0.7782 / 123.73`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Mid-band weight sweep on the same `24..72 delayed_only` window:

- `14311` `lw=0.1125`: `0.6514 / 0.3595 / 0.7895 / 124.00`
- `14312` `lw=0.1250`: `0.6509 / 0.3793 / 0.7934 / 124.48`
- `14313` `lw=0.1375`: `0.6626 / 0.4002 / 0.7994 / 124.49`

Interpretation:

- `14311` is a clear regression on both content and route
- `14312` recovers some of the lost route but still trails both `13711` and
  `14213`
- `14313` is the only real challenger: it slightly beats `13711` on selected
  `overall` and `fq_acc`, nearly matches `13711` route, and improves on
  `14213` route

Rerun:

- `14313_rerun1` matched exactly:
  `0.6626 / 0.4002 / 0.7994 / 124.49`

Confirm:

- `14313` base confirm: `0.6414 / 0.3758 / 0.7734 / 123.68`
- `14313` full_locked: `0.1306 / 0.2539 / 0.9876 / 126.43`
- `14313` finalquery_heavy: `0.1980 / 0.2465 / 0.9860 / 126.44`
- `14313` longdistance: `0.1846 / 0.2635 / 0.9843 / 157.96`

Conclusion:

- the late-start delayed-only optimum really is narrow: `0.1375` reproduced
  exactly and is meaningfully better than the weaker mid-band weights on the
  dev selector
- but the confirm pass is decisive again: held-confirm behavior is unchanged
  from `13711` and `14213`
- base confirm keeps a little more final-query content than `13711`, but it
  still gives back too much route to count as a promotion

So `gnn2-hqa` closes as a rerun-backed mixed result rather than a confirmed
improvement.
