## gnn2-nfu

Question: what caused the original rank-1 failure in the long
`hybrid_es_benchmark_b_v2_maskcurr_h256` main run, and was it a distributed
runtime problem, a PyTorch nightly issue, or a model-side failure?

Primary evidence:

- The original run directory
  [hybrid_es_b_v2_maskcurr_h256](/home/catid/gnn2/results/phase2_main/hybrid_es_b_v2_maskcurr_h256)
  never wrote a `summary.json` or `hybrid_es_best.pt`.
- Its `metrics.jsonl` contains only `warmstart` entries:
  - `145` warmstart train steps were logged, ending at step `144`
  - only `3` warmstart validation checkpoints were emitted: steps `39`, `79`,
    and `119`
  - there are **no** `hybrid_es` generation records at all
- The stable rerun
  [hybrid_es_b_v2_maskcurr_h256_stable](/home/catid/gnn2/results/phase2_main/hybrid_es_b_v2_maskcurr_h256_stable)
  completed after reducing only the budget:
  - warmstart `220 -> 140`
  - ES generations `72 -> 56`
- Later repo documentation already identifies the exact failure mode:
  - [phase4_report.md](/home/catid/gnn2/docs/phase4_report.md) says only rank 0
    executed warmstart while rank 1 waited at a distributed barrier, and the
    default 10-minute process-group timeout could kill the job before ES started
  - [phase4_lessons.md](/home/catid/gnn2/docs/phase4_lessons.md) records the
    same bug and says phase 4 fixed it by adding an explicit distributed timeout
- The current code now matches that diagnosis:
  - [run.py](/home/catid/gnn2/src/train/run.py#L116) sets
    `timeout=timedelta(minutes=GNN2_DIST_TIMEOUT_MINUTES)` in
    `dist.init_process_group()`

Interpretation:

- The original `phase2_main` run did **not** fail because hybrid ES itself hit a
  model-side numerical bug on rank 1.
- It also does **not** point to a PyTorch nightly regression as the primary
  cause.
- The failure is most consistent with a distributed warmstart-barrier timeout:
  rank 1 waited in the process group while rank 0 spent too long in the
  single-rank warmstart phase, then `torchrun` terminated the whole job before
  the first ES generation.

Conclusion:

- `gnn2-nfu` closes as a diagnosed systems bug
- root cause: distributed timeout during single-rank warmstart
- not the model, not a late ES-rank numerical failure
- the shorter stability rerun and the later explicit timeout fix are both
  consistent with that diagnosis
