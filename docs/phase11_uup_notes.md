## gnn2-uup

Question: if the delayed-only `13111` source-logit signal helps mainly at the start of training, can stopping it early improve the stable `13411` frozen-`1201` boundary without giving back the imported route structure?

Baseline references:

- `13411` selected val: `0.6523 / 0.3833 / 0.7766 / 124.08`
- `13411` base confirm: `0.6479 / 0.3826 / 0.7884 / 123.97`
- `13612` selected val: `0.6650 / 0.3893 / 0.7855 / 124.19`
- `13612` base confirm: `0.6399 / 0.3598 / 0.7855 / 124.19`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Schedule sweep:

- `13711` stop@48: `0.6567 / 0.3923 / 0.8014 / 123.76`
- `13712` stop@96: `0.6650 / 0.3764 / 0.7786 / 123.52`
- `13713` stop@192: `0.6401 / 0.3496 / 0.7666 / 123.85`
- `13714` decay `1.0 -> 0.0` over `0..192`: `0.6548 / 0.3615 / 0.7666 / 123.50`

`13711` was the only clear promotion candidate. It improved selected-val final-query accuracy and route over both `13411` and `13612` without collapsing overall behavior.

Rerun:

- `13711_rerun1` matched exactly at selected step `239`
- rerun val: `0.6567 / 0.3923 / 0.8014 / 123.76`

Confirm:

- `13711` base confirm: `0.6392 / 0.3705 / 0.8024 / 124.42`
- `13711` full_locked: `0.1306 / 0.2539 / 0.9876 / 126.43`
- `13711` finalquery_heavy: `0.1980 / 0.2465 / 0.9860 / 126.44`
- `13711` longdistance: `0.1846 / 0.2635 / 0.9843 / 157.96`

Conclusion:

- early shutoff is a real and reproducible schedule effect on the dev selector
- the best short-stop schedule increases route fidelity materially
- but confirmed base content stays below `13411`
- and held-confirm behavior is unchanged from the existing `13612` plateau

So `uup` closes as a rerun-backed mixed result, not a promoted breakthrough.
