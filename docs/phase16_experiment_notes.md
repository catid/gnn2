# Phase 16 Experiment Notes

## 2026-03-26: Trajectory-Token Sidecar Retrieval

Question:
Can the sidecar path improve content recovery if it retrieves over actual
trajectory-bank tokens instead of first collapsing trajectory state to a single
summary vector?

Implementation:
- Added `factorized_content_sidecar_mode: trajectory_kv_memory`.
- The new mode is route-isolated by construction:
  - routing, control, and exit logic stay frozen
  - sidecar source must be `trajectory_bank`
  - sidecar writes/reads affect only the final content readout path
- Retrieval uses token-level trajectory-bank slots as keys/values with a
  query-conditioned attention read.

Validation:
- Added config guard coverage and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'sidecar or slot'`
  passed (`5 passed, 25 deselected`).
- `uv run pytest -q` passed (`82 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajtokens_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajtokens_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260326_235713_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajtokens_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260326_235713_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajtokens_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9971 / 0.9950 / 0.9345 / 120.80`
  - `full_locked`: `0.9985 / 0.9981 / 0.9477 / 122.07`
  - `finalquery_heavy`: `0.9985 / 0.9982 / 0.9430 / 121.35`
  - `longdistance`: `0.9961 / 0.9951 / 0.9453 / 152.41`
  in `overall / fq_acc / fq_route / fq_exit` order.

Hard-slice comparison versus the stronger phase-15 sidecar baseline `18052`:
- Artifact:
  [18052_vs_phase16_trajtokens_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/18052_vs_phase16_trajtokens_confirm32b.json)
- Result:
  - baseline beat candidate on late-route disagreements `3-2`
  - baseline `late_wrong_content = 2`
  - candidate `late_wrong_content = 3`

Conclusion:
- The architecture is safe: it does not reopen routing instability and preserves
  the expected late-route regime.
- The architecture is not a validated improvement over the stronger phase-15
  sidecar baseline.
- Direct token-level trajectory retrieval alone does not appear to solve the
  remaining bottleneck.

Next architectural direction:
- Focus on sidecar writing/addressing quality rather than slot exposure alone.
- The next justified follow-up is a learned sparse write-address or write-gated
  sidecar that decides which trajectory states enter sidecar memory.
