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

## 2026-03-27: Head-Diverse Multi-Head Trajectory Writer

Question:
If the multi-head content-write-value branch still ties the simpler phase-16
leaders, can we force real head specialization by giving each head its own
projected source view of trajectory-bank states before sparse writing?

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_headwise_write_value_gated_kv_memory`.
- This keeps the route-isolated multi-head content-write-value branch, but each
  write head now scores trajectory-bank states through its own projected source
  view before top-k sparse selection.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new head-specific source projection only affects the final content
    readout path

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`17 passed, 25 deselected`).
- `uv run pytest -q` passed (`94 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadheadwisewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadheadwisewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_021302_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadheadwisewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_021302_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadheadwisewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9961 / 0.9930 / 0.9424 / 121.91`
  - `full_locked`: `0.9951 / 0.9925 / 0.9281 / 120.27`
  - `finalquery_heavy`: `0.9985 / 0.9982 / 0.9492 / 122.01`
  - `longdistance`: `0.9961 / 0.9945 / 0.9390 / 152.15`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.693 / 0.506`
  - write entropy/top1: `0.687 / 0.532`
  - value gate mean: `0.504`

Hard-slice comparisons:
- Versus current phase-16 multi-head content-write-value branch:
  [phase16_multiheadcontentwritevalue_vs_phase16_headdiverse_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_headdiverse_confirm32b.json)
  - baseline beat candidate `3-1` on late-route disagreements
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus phase-16 plain write-gated baseline:
  [phase16_writegate_vs_phase16_headdiverse_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_headdiverse_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus phase-16 content-write baseline:
  [phase16_contentwrite_vs_phase16_headdiverse_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwrite_vs_phase16_headdiverse_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`

Conclusion:
- Learned head-specific source views are route-safe, but they are a real
  quality regression.
- Simply parameterizing the heads separately is not enough; it softens the
  selected slices and worsens confirm-time wrong-content behavior.
- The next justified move, if we continue in this family, is explicit disjoint
  or budgeted write allocation across heads rather than another learned
  reparameterization of the same dense source bank.

## 2026-03-27: Disjoint-Write Multi-Head Trajectory Sidecar

Question:
If learned head-specific source views made the multi-head writer worse, can we
recover useful specialization by assigning heads to disjoint trajectory-bank
slots directly while keeping the content-conditioned value gate and route
isolation contract intact?

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_disjoint_write_value_gated_kv_memory`.
- This reuses the stable multi-head content-write-value sidecar, but replaces
  independent per-head top-k writes with explicit greedy disjoint assignment
  across heads, plus a fallback assignment for heads that would otherwise get no
  slot.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new logic only changes how the content-only sidecar selects write slots

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`19 passed, 25 deselected`).
- `uv run pytest -q` passed (`96 passed`).
- The first training attempt exposed an autograd error from in-place assignment
  in the disjoint writer path; that was fixed by rebuilding the assignment mask
  and assignment values without mutating views needed for backward.

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheaddisjointwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheaddisjointwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_023655_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheaddisjointwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_023655_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheaddisjointwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9985 / 0.9980 / 0.9454 / 121.80`
  - `full_locked`: `0.9990 / 0.9991 / 0.9496 / 122.27`
  - `finalquery_heavy`: `0.9980 / 0.9976 / 0.9388 / 121.08`
  - `longdistance`: `0.9956 / 0.9938 / 0.9293 / 150.56`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.689 / 0.531`
  - write entropy/top1: `0.682 / 0.542`
  - value gate mean: `0.510`

Hard-slice comparisons:
- Versus current phase-16 multi-head content-write-value branch:
  [phase16_multiheadcontentwritevalue_vs_phase16_disjoint_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_disjoint_confirm32b.json)
  - baseline beat candidate `3-1` on late-route disagreements
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus phase-16 plain write-gated baseline:
  [phase16_writegate_vs_phase16_disjoint_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_disjoint_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus phase-16 content-write baseline:
  [phase16_contentwrite_vs_phase16_disjoint_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwrite_vs_phase16_disjoint_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`

Conclusion:
- Explicit disjoint write allocation is route-safe, but it is a confirm-time
  regression despite strong summary slices.
- Forcing hard disjointness appears to starve the writer of the shared fallback
  behavior that the better phase-16 sidecars still exploit on the decisive
  held-confirm content failures.
- The next justified architecture move is a hybrid writer that preserves head
  diversity without enforcing full slot disjointness, such as reserved
  per-head budgets plus a shared fallback slot pool.

## 2026-03-27: Reserved-Fallback Multi-Head Trajectory Sidecar

Question:
If hard disjoint write allocation is too restrictive, can a hybrid writer keep
one reserved per-head slot for diversity while letting the remaining write
budget come from a shared fallback pool and recover the confirm-time losses?

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_reserved_fallback_write_value_gated_kv_memory`.
- Each write head first claims one reserved slot from a greedy non-overlapping
  budget. The remaining write budget is then drawn from a shared fallback pool
  that excludes only the head's own reserved slot, so heads can still overlap on
  useful high-value states without collapsing back to fully shared behavior.
- The content-conditioned value gate is preserved.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new logic only changes content-sidecar write selection

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`21 passed, 25 deselected`).
- `uv run pytest -q` passed (`98 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedfallbackwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedfallbackwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_025917_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedfallbackwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_025917_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedfallbackwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9985 / 0.9980 / 0.9325 / 120.42`
  - `full_locked`: `0.9985 / 0.9981 / 0.9402 / 121.52`
  - `finalquery_heavy`: `0.9990 / 0.9994 / 0.9430 / 121.61`
  - `longdistance`: `0.9990 / 0.9986 / 0.9543 / 153.54`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.693 / 0.502`
  - write entropy/top1: `0.659 / 0.598`
  - value gate mean: `0.510`

Hard-slice comparisons:
- Versus current phase-16 multi-head content-write-value branch:
  [phase16_multiheadcontentwritevalue_vs_phase16_reservedfallback_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_reservedfallback_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 plain write-gated baseline:
  [phase16_writegate_vs_phase16_reservedfallback_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_reservedfallback_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus phase-16 content-write baseline:
  [phase16_contentwrite_vs_phase16_reservedfallback_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_contentwrite_vs_phase16_reservedfallback_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`

Conclusion:
- The reserved-fallback hybrid fixes the confirm-time regression from the
  disjoint writer without reopening routing.
- It does not clearly beat the current phase-16 leaders on the decisive
  hard-slice gate, but it is the first hybrid writer in this branch to match
  them there while also improving aggregate confirm accuracy over the simpler
  write-gated and content-write variants.
- The next justified step is not another bounded architecture scout. This
  branch is promising enough to merit exact rerun plus locked confirm.

## 2026-03-27: Reserved-Mix Trajectory Sidecar

Question:
Can the reserved-fallback writer improve if each head learns an explicit
reserved-versus-shared mix, rather than relying on a single softmax over the
reserved slot plus shared fallback slots?

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_reserved_mixed_write_value_gated_kv_memory`.
- This keeps the reserved-fallback slot allocation, then computes separate
  reserved and shared write distributions and mixes them with a learned
  per-head content-conditioned gate.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new logic changes only content-sidecar write weighting

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`25 passed, 25 deselected` after the follow-on additions in this file).
- `uv run pytest -q` passed (`102 passed` after the follow-on additions in this file).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedmixwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedmixwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_032021_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedmixwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_032021_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedmixwritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9985 / 0.9990 / 0.9355 / 120.46`
  - `full_locked`: `0.9976 / 0.9963 / 0.9449 / 121.77`
  - `finalquery_heavy`: `0.9976 / 0.9969 / 0.9430 / 121.76`
  - `longdistance`: `0.9985 / 0.9979 / 0.9439 / 152.38`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - read entropy/top1: `0.692 / 0.511`
  - write entropy/top1: `0.688 / 0.543`
  - value gate mean: `0.505`

Hard-slice comparisons:
- Versus reserved-fallback:
  [phase16_reservedfallback_vs_phase16_reservedmix_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_reservedfallback_vs_phase16_reservedmix_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus multi-head content-write-value:
  [phase16_multiheadcontentwritevalue_vs_phase16_reservedmix_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_reservedmix_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus plain write-gated:
  [phase16_writegate_vs_phase16_reservedmix_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_reservedmix_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`

Conclusion:
- The learned reserved-versus-shared mix is a confirm-time regression.
- The mixture gate makes the writer more flexible in principle, but on the
  decisive confirm slice it softens the stronger reserved-fallback behavior and
  reintroduces wrong-content errors.
- The next justified architecture move is to leave reserved and shared paths
  separate and only calibrate the reserved competition itself.

## 2026-03-27: Reserved-Temperature Trajectory Sidecar

Question:
If reserved-fallback is the current hybrid leader, can it be improved by
leaving the shared fallback pool untouched and only sharpening or flattening the
reserved-slot competition with a learned per-head content-conditioned
temperature?

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_reserved_temperature_write_value_gated_kv_memory`.
- This keeps the reserved-fallback writer structure, but applies a learned
  per-head reserved temperature before the final write softmax. Shared fallback
  logits are left unchanged.
- Added `factorized_content_sidecar_reserved_temperature_mean` for bounded
  instrumentation.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new logic changes only the content-sidecar write scores

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`25 passed, 25 deselected`).
- `uv run pytest -q` passed (`102 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedtemperaturewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedtemperaturewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_034212_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedtemperaturewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_034212_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedtemperaturewritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9971 / 0.9950 / 0.9345 / 120.80`
  - `full_locked`: `0.9995 / 0.9991 / 0.9477 / 121.90`
  - `finalquery_heavy`: `0.9980 / 0.9976 / 0.9424 / 121.40`
  - `longdistance`: `0.9961 / 0.9945 / 0.9446 / 152.23`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - write entropy/top1: `0.692 / 0.516`
  - value gate mean: `0.500`
  - reserved temperature mean: `1.190`

Hard-slice comparisons:
- Versus reserved-fallback:
  [phase16_reservedfallback_vs_phase16_reservedtemperature_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_reservedfallback_vs_phase16_reservedtemperature_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus multi-head content-write-value:
  [phase16_multiheadcontentwritevalue_vs_phase16_reservedtemperature_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_reservedtemperature_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus plain write-gated:
  [phase16_writegate_vs_phase16_reservedtemperature_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_reservedtemperature_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`

Conclusion:
- Calibrating reserved-slot sharpness alone is not enough. This branch looks
  excellent on selected summary slices but is another confirm-time false
  positive.
- The negative result is informative: the remaining bottleneck is not simple
  reserved-path softness. The stronger reserved-fallback baseline is already in
  the right regime, and changing reserved calibration without improving shared
  fallback selection makes confirm-time content worse.
- The next justified architecture direction is to keep the reserved-fallback
  scaffold fixed and instead change shared fallback pressure or admission, not
  reserved-slot temperature.

## 2026-03-27: Shared-Penalty Reserved-Fallback Trajectory Sidecar

Question:
If the reserved-fallback scaffold is the current hybrid leader, can it be
improved by making shared fallback admission more selective without touching the
reserved path? The bounded test here applies a learned per-head penalty only to
slots that are already reserved by another head before shared fallback top-k
selection.

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_reserved_shared_penalty_write_value_gated_kv_memory`.
- This keeps the reserved-fallback writer structure:
  - one reserved slot per head from a greedy non-overlapping budget
  - shared fallback slots selected afterward
- The only change is a content-conditioned per-head penalty applied to slots
  already reserved by another head during the shared fallback stage.
- Added `factorized_content_sidecar_shared_penalty_mean` for bounded tracing.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new logic changes only content-sidecar write selection

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`27 passed, 25 deselected`).
- `uv run pytest -q` passed (`104 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedsharedpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedsharedpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_040414_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedsharedpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_040414_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedsharedpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9976 / 0.9960 / 0.9434 / 121.25`
  - `full_locked`: `0.9980 / 0.9981 / 0.9468 / 121.75`
  - `finalquery_heavy`: `0.9985 / 0.9982 / 0.9461 / 121.91`
  - `longdistance`: `0.9966 / 0.9951 / 0.9446 / 152.89`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - write entropy/top1: `0.687 / 0.530`
  - value gate mean: `0.504`
  - shared penalty mean: `0.962`

Hard-slice comparisons:
- Versus reserved-fallback:
  [phase16_reservedfallback_vs_phase16_reservedsharedpenalty_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_reservedfallback_vs_phase16_reservedsharedpenalty_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus multi-head content-write-value:
  [phase16_multiheadcontentwritevalue_vs_phase16_reservedsharedpenalty_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_reservedsharedpenalty_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`
- Versus plain write-gated:
  [phase16_writegate_vs_phase16_reservedsharedpenalty_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_reservedsharedpenalty_confirm32b.json)
  - late-route disagreements split `1-1`
  - both runs kept `late_wrong_content = 1`

Conclusion:
- This is a bounded positive, not a breakthrough.
- Unlike reserved-mix and reserved-temperature, changing shared fallback
  admission does not regress the decisive confirm slice.
- The branch now matches the current phase-16 leaders on the hard-slice gate
  while preserving `late_wrong_content = 1`, so the justified next step is
  exact rerun plus locked confirm rather than another immediate bounded scout.

## 2026-03-27: Reservation-Strength Shared-Penalty Sidecar

Question:
Can the shared-penalty reserved-fallback writer be improved by scaling the
shared fallback penalty by how strongly another head reserved a slot, instead
of using a flat per-head penalty? The intended benefit was to keep strongly
claimed reserved slots cleaner while still allowing fallback access to weaker
claims.

Implementation:
- Added
  `factorized_content_sidecar_mode: trajectory_content_multihead_reserved_strength_penalty_write_value_gated_kv_memory`.
- This keeps the same route-isolated reserved-fallback scaffold:
  - one greedy non-overlapping reserved slot per head
  - shared fallback slots selected afterward
- The only architectural change is in the shared fallback stage:
  - compute a per-slot reservation strength from the strongest reserved score
  - multiply that by a learned per-head penalty
  - subtract the product before fallback top-k selection
- Added
  `factorized_content_sidecar_reserved_strength_penalty_mean`
  for bounded tracing.
- The route-isolation contract still holds:
  - routing, control, and exit remain frozen
  - sidecar source must be `trajectory_bank`
  - the new logic changes only content-sidecar fallback write selection

Validation:
- Added config-guard and zero-init route-isolation tests in
  [tests/test_routing_semantics.py](/home/catid/gnn2/tests/test_routing_semantics.py).
- `uv run pytest -q tests/test_routing_semantics.py -k 'trajectory or sidecar or slot'`
  passed (`29 passed, 25 deselected`).
- `uv run pytest -q` passed (`106 passed`).

Bounded dev run:
- Config:
  [hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedstrengthpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml](/home/catid/gnn2/configs/phase16/dev/hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedstrengthpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi.yaml)
- Result dir:
  [20260327_042722_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedstrengthpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi](/home/catid/gnn2/results/phase16_dev/20260327_042722_hard_st_benchmark_b_v2_teacher1874_contentpath_resume16045_sidecartrajmultiheadreservedstrengthpenaltywritevalue_teacher16081_contentmse010_hardslice_fqhld_selectlexi)
- Summary slices:
  - `best_val`: `0.9980 / 0.9970 / 0.9404 / 121.46`
  - `full_locked`: `0.9980 / 0.9981 / 0.9393 / 121.26`
  - `finalquery_heavy`: `0.9985 / 0.9982 / 0.9357 / 121.01`
  - `longdistance`: `0.9971 / 0.9958 / 0.9480 / 153.18`
  in `overall / fq_acc / fq_route / fq_exit` order.
- Sidecar usage on selected `full_locked` slice:
  - write entropy/top1: `0.689 / 0.524`
  - value gate mean: `0.500`
  - reserved strength penalty mean: `1.008`

Hard-slice comparisons:
- Versus reserved-fallback:
  [phase16_reservedfallback_vs_phase16_reservedstrengthpenalty_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_reservedfallback_vs_phase16_reservedstrengthpenalty_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus multi-head content-write-value:
  [phase16_multiheadcontentwritevalue_vs_phase16_reservedstrengthpenalty_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_multiheadcontentwritevalue_vs_phase16_reservedstrengthpenalty_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`
- Versus plain write-gated:
  [phase16_writegate_vs_phase16_reservedstrengthpenalty_confirm32b.json](/home/catid/gnn2/artifacts/phase15_hardslice/phase16_writegate_vs_phase16_reservedstrengthpenalty_confirm32b.json)
  - baseline beat candidate `3-1`
  - candidate `late_wrong_content` worsened `1 -> 3`

Conclusion:
- This is a verified negative.
- Strength-weighting the fallback penalty makes the shared stage too brittle:
  summary slices stay strong, but confirm-time late-route content gets worse.
- The failure is informative because it narrows the next move:
  the shared fallback stage should reallocate or prioritize leftover capacity,
  not scale a stronger anti-sharing penalty from reserved strength.
