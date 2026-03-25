# Phase 12 Handoff

- Best confirmed trajectory-aware reader result:
  the strongest exact-rerun-clean trajectory-aware family is
  [15051](/home/catid/gnn2/results/phase12_dev/hard_st_b_v2_teacher1874_temporalbank_sinkreadout_bilinear_exit_routehist_seed15051_p1),
  with base `0.9850 / 0.9699 / 0.9505 / 122.40` and full_locked
  `0.6494 / 0.3159 / 0.8771 / 115.49` for
  `overall / fq_acc / fq_route / fq_exit`.
- Best confirmed secondary-source portability result:
  there is no robust strong transfer. The completed portability panels settled
  near base overall `0.9646`, route `0.8274`, exit `80.43` on `1821` and base
  overall `0.9462`, route `0.8253`, exit `80.54` on `1842`.
- Whether route-trace conditioning helped:
  no. It was mostly neutral to harmful. The strongest phase-12 readers were
  route-blind.
- Whether contiguous windows beat endpoint-only readers:
  not as a general rule. The old `1201` contiguous-window clue did not
  transfer into a broad reader advantage.
- Whether a sink change was needed:
  no. Minimal keyed sinks were tested and retired because they created stable
  early-exit shortcuts instead of rescuing held-confirm content.
- Whether ES helped at the reader level:
  phase 12 did not enter a serious ES branch. Cluster G was left correctly
  unused because no competitive reader frontier justified it.
- Which clusters were retired and why:
  Cluster A retired as a strong mapping result because temporal banks help base
  behavior but not held confirms; Cluster B retired as a strong mapping result
  because factorized readers help base behavior but not held confirms; Cluster C
  retired because contiguous-window and portability claims collapsed under fair
  panels; Cluster D retired as a rerun-backed wrong-regime negative; Cluster E
  retired as a serious adapter negative; Cluster F retired because keyed sinks
  yield stable off-regime shortcuts.
- Single next experiment:
  a narrow stability-focused `1874` experiment on the best route-blind
  temporal/factorized reader, explicitly regularized against the early-exit
  shortcut basin while preserving the late-route solution.
