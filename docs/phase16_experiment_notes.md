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

## 2026-03-27: Write-Gated Trajectory Sidecar

Question:
Does explicit sparse write selection over trajectory-bank states help more than
 exposing all trajectory tokens directly to sidecar retrieval?

Implementation:
- Added `factorized_content_sidecar_mode: trajectory_write_gated_kv_memory`.
- The new mode stays route-isolated:
  - routing, control, and exit behavior remain frozen
  - sidecar source must be `trajectory_bank`
  - a learned write-query selects a sparse top-k subset of trajectory-bank
    states before the normal sidecar retrieval step
- Added write-gate trace/metric exposure:
  - `factorized_content_sidecar_write_weights`
  - `factorized_content_sidecar_write_entropy`
  - `factorized_content_sidecar_write_top1_weight`

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`7 passed, 25 deselected`).
- `uv run pytest -q` passed (`84 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajwritegate_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajwritegate_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_002025_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajwritegate_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_002025_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajwritegate_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9980 / 0.9970 / 0.9424 / 121.63`
  - `full_locked`: `0.9985 / 0.9972 / 0.9393 / 121.17`
  - `finalquery_heavy`: `0.9980 / 0.9976 / 0.9430 / 121.60`
  - `longdistance`: `0.9990 / 0.9986 / 0.9466 / 152.99`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.661 / 0.593`
  - write entropy/top1: `0.680 / 0.560`

Hard-slice comparison versus stronger phase-15 sidecar baseline `18052`:
- Artifact:
  [18052_vs_phase16_writegate_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/18052_vs_phase16_writegate_confirm32b.json)
- Result:
  - candidate beat baseline on late-route disagreements `2-1`
  - baseline `late_wrong_content = 2`
  - candidate `late_wrong_content = 1`

Conclusion:
- This is the first bounded positive in the phase-16 sidecar-writing line.
- Explicit sparse write selection looks more useful than exposing raw
  trajectory-bank tokens directly.
- The next justified step is to rerun and run locked confirm on this write-gated
  branch before making a stronger claim.

## 2026-03-27: Content-Conditioned Write-Gated Trajectory Sidecar

Question:
Does sparse write selection improve further if the write address can also see
the current factorized content representation, not just query and route
features?

Implementation:
- Added `factorized_content_sidecar_mode: trajectory_content_write_gated_kv_memory`.
- This keeps the same route-isolated trajectory sidecar as the write-gated
  branch, but the write query now includes:
  - projected query hidden state
  - projected route features from the trajectory-bank anchor path
  - projected current factorized content hidden state
- The sidecar still only affects final content readout.

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`9 passed, 25 deselected`).
- `uv run pytest -q` passed (`86 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwrite_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwrite_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_004142_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwrite_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_004142_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwrite_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9990 / 0.9990 / 0.9374 / 120.79`
  - `full_locked`: `0.9980 / 0.9981 / 0.9412 / 121.58`
  - `finalquery_heavy`: `0.9980 / 0.9976 / 0.9437 / 121.67`
  - `longdistance`: `0.9995 / 0.9993 / 0.9487 / 153.03`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.686 / 0.543`
  - write entropy/top1: `0.677 / 0.557`

Hard-slice comparisons:
- Versus current phase-16 write-gated baseline:
  [phase16_writegate_vs_phase16_contentwrite_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_contentwrite_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus stronger phase-15 sidecar baseline `18052`:
  [18052_vs_phase16_contentwrite_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/18052_vs_phase16_contentwrite_confirm32b.json)
  - candidate beat baseline on late-route disagreements `2-1`
  - baseline `late_wrong_content = 2`
  - candidate `late_wrong_content = 1`

Conclusion:
- Content-conditioned write addressing is safe and remains in the stable
  late-route regime.
- It improves over the older phase-15 sidecar baseline.
- It does not yet cleanly separate from the simpler phase-16 write-gated
  variant, so the justified next step is a rerun plus locked-confirm head-to-head
  between the two write-gated variants, not immediate promotion.

## 2026-03-27: Content-Conditioned Write-Value Gated Trajectory Sidecar

Question:
If write addressing is already content-conditioned, does adding a
content-conditioned value gate improve what gets preserved from each selected
trajectory state?

Implementation:
- Added `factorized_content_sidecar_mode: trajectory_content_write_value_gated_kv_memory`.
- This extends the content-conditioned write-gated trajectory sidecar by adding
  a content-conditioned gate over the projected value slots before readout.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new value gate affects only the final content readout path
- Added `factorized_content_sidecar_value_gate_mean` for bounded usage tracing.

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`11 passed, 25 deselected`).
- `uv run pytest -q` passed (`88 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_010602_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_010602_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9990 / 1.0000 / 0.9553 / 122.51`
  - `full_locked`: `0.9985 / 0.9981 / 0.9356 / 120.82`
  - `finalquery_heavy`: `0.9980 / 0.9976 / 0.9412 / 121.31`
  - `longdistance`: `0.9971 / 0.9958 / 0.9515 / 153.70`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.688 / 0.535`
  - write entropy/top1: `0.679 / 0.560`
  - value gate mean: `0.504`

Hard-slice comparisons:
- Versus phase-16 content-write baseline:
  [phase16_contentwrite_vs_phase16_contentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwrite_vs_phase16_contentwritevalue_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 plain write-gated baseline:
  [phase16_writegate_vs_phase16_contentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_contentwritevalue_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus stronger phase-15 sidecar baseline `18052`:
  [18052_vs_phase16_contentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/18052_vs_phase16_contentwritevalue_confirm32b.json)
  - candidate beat baseline on late-route disagreements `2-1`
  - baseline `late_wrong_content = 2`
  - candidate `late_wrong_content = 1`

Conclusion:
- Content-conditioned value gating is safe and weakly positive over the older
  phase-15 sidecar baseline.
- It does not cleanly improve over the current phase-16 write-gated family.
- The current map suggests that simple value gating alone is not the missing
  ingredient; the next justified move is still rerun/confirm ranking of the
  leading write-gated variants, or a more structural write mechanism such as
  multi-head sparse writing.

## 2026-03-27: Multi-Head Sparse Trajectory Writer

Question:
If scalar content/write gating is no longer enough to separate the leading
phase-16 sidecar variants, does a structurally richer writer help? The bounded
test here was a small multi-head sparse writer over trajectory-bank states,
still isolated from routing and used only by the content readout path.

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_write_gated_kv_memory`.
- This mode forms a small set of head-specific memory slots by running multiple
  sparse write heads over trajectory-bank states, then lets the final sidecar
  read select over those head summaries.
- Added `factorized_content_sidecar_write_heads` to control the number of write
  heads while preserving the existing top-k sparse write contract.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new multi-head writer only affects the final content readout path

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`13 passed, 25 deselected`).
- `uv run pytest -q` passed (`90 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultihead_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultihead_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_012826_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultihead_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_012826_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultihead_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9971 / 0.9960 / 0.9275 / 120.15`
  - `full_locked`: `1.0000 / 1.0000 / 0.9468 / 121.74`
  - `finalquery_heavy`: `0.9980 / 0.9976 / 0.9388 / 120.88`
  - `longdistance`: `0.9980 / 0.9972 / 0.9453 / 152.51`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.693 / 0.502`
  - write entropy/top1: `0.679 / 0.562`

Hard-slice comparisons:
- Versus phase-16 content-write baseline:
  [phase16_contentwrite_vs_phase16_multihead_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwrite_vs_phase16_multihead_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 plain write-gated baseline:
  [phase16_writegate_vs_phase16_multihead_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_multihead_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus stronger phase-15 sidecar baseline `18052`:
  [18052_vs_phase16_multihead_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/18052_vs_phase16_multihead_confirm32b.json)
  - candidate beat baseline on late-route disagreements `2-1`
  - baseline `late_wrong_content = 2`
  - candidate `late_wrong_content = 1`

Conclusion:
- Multi-head sparse writing is safe and improves over the older phase-15
  sidecar baseline.
- It does not cleanly improve over the current phase-16 write-gated family.
- The current map suggests that richer write structure alone is not enough; the
  next justified architecture move is to combine multi-head sparse writing with
  content-conditioned value preservation rather than treating those mechanisms
  separately.

## 2026-03-27: Multi-Head Content-Write-Value Trajectory Sidecar

Question:
If multi-head sparse writing and content-conditioned value preservation both
look safe but only tie the phase-16 write-gated leaders on the confirm
hard-slice, does combining them produce a real separation?

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_write_value_gated_kv_memory`.
- This mode keeps the route-isolated multi-head sparse trajectory writer, then
  applies the same content-conditioned value gate used by the simpler
  content-write-value sidecar before final readout.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new branch only affects the final content readout path

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`15 passed, 25 deselected`).
- `uv run pytest -q` passed (`92 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_015203_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_015203_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadcontentwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9995 / 1.0000 / 0.9305 / 120.16`
  - `full_locked`: `1.0000 / 1.0000 / 0.9542 / 122.66`
  - `finalquery_heavy`: `0.9990 / 0.9994 / 0.9467 / 121.75`
  - `longdistance`: `0.9980 / 0.9972 / 0.9473 / 152.72`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.693 / 0.502`
  - write entropy/top1: `0.659 / 0.591`
  - value gate mean: `0.508`

Hard-slice comparisons:
- Versus phase-16 plain write-gated baseline:
  [phase16_writegate_vs_phase16_multiheadcontentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_multiheadcontentwritevalue_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 content-write baseline:
  [phase16_contentwrite_vs_phase16_multiheadcontentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwrite_vs_phase16_multiheadcontentwritevalue_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 multi-head baseline:
  [phase16_multihead_vs_phase16_multiheadcontentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multihead_vs_phase16_multiheadcontentwritevalue_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 content-write-value baseline:
  [phase16_contentwritevalue_vs_phase16_multiheadcontentwritevalue_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwritevalue_vs_phase16_multiheadcontentwritevalue_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`

Conclusion:
- The combined multi-head content-write-value branch is safe and improves
  aggregate confirm accuracy over the earlier phase-16 variants.
- It does not cleanly separate on the actual confirm hard-slice gate; it ties
  all serious phase-16 comparators `1-1` on late-route disagreements.
- The current map suggests the missing ingredient is not just combining existing
  safe mechanisms. The next justified architecture move is to force more
  diversity or specialization across write heads, rather than letting the heads
  behave as near-interchangeable sparse selectors.
