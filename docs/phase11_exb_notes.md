## gnn2-exb

Question: after `hqa` found a real late-start delayed-only optimum at `14313`,
is the remaining gap mostly a checkpoint-selection problem rather than a
training problem?

Baseline references:

- `14313` selected val: `0.6626 / 0.4002 / 0.7994 / 124.49`
- `14313` base confirm: `0.6414 / 0.3758 / 0.7734 / 123.68`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Selector-only sweep on the exact `14313` training setup:

- `14411` lexicographic route floor `0.795`:
  `0.6592 / 0.3853 / 0.7845 / 123.23`
- `14412` lexicographic route floor `0.80`:
  `0.6318 / 0.3337 / 0.7597 / 123.42`
- `14413` weighted geomean over base fq_acc plus route:
  `0.6597 / 0.3813 / 0.7716 / 124.28`

Interpretation:

- the softer lexicographic floor at `0.795` already underperforms the plain
  `14313` selector on both content and route
- tightening the floor to `0.80` is much worse and clearly overconstrains the
  selection
- the weighted geomean also fails to recover the `14313` content gain while
  giving back even more route

Conclusion:

- the `14313` training trace does not appear to hide a better checkpoint that
  can be recovered by simple route-aware selector changes
- this makes the map sharper: the remaining loss is not just the checkpoint
  rule on this branch

So `gnn2-exb` closes as a fair local negative with no rerun candidate.
