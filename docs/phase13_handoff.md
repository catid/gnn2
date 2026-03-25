# Phase 13 Handoff

Phase 13 is closed.

- Best current confirmed stable `1874` result:
  [16045](/home/catid/gnn2/results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_15051readout_15057extras_selectlocked_seed16045_p1)
  is still the best fully paneled stable bridge at locked panel mean
  `0.9965 / 0.9460 / 121.86`, and
  [16066](/home/catid/gnn2/results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_noroute_paramanchor15057_w0005_selectlexi_lockedroute_lockedfqacc_stop720_seed16066_p1)
  is now the cleanest completed Cluster A family after its five-seed panel
  finished at locked mean `0.9557 / 0.9391 / 121.34`.
- Whether `15057`-style upside can be stabilized:
  partly yes. Conservative continuation from `15051` into the `15057`-style
  reader survives a five-seed panel without recreating the old shortcut
  collapse.
- Whether continuation from `15051` helped:
  yes. It is the clearest positive result of phase 13 so far.
- Whether hard-case mining or held-confirm-aware selection helped:
  only at summary time. Those gains have not survived independent confirm.
- Current verification read:
  the `16066` sustained-anchor panel is complete and cleaner than `16022`.
  `16066` finished with four strong late-route seeds plus one weaker content
  dip for locked mean `0.9557 / 0.9391 / 121.34`, while `16022` finished as a
  mixed high-variance panel at `0.8530 / 0.9399 / 121.29` with one severe
  late-route content-collapse outlier and one middling recovery seed.
  The extra `16041` bridge panel is now also complete at
  `0.9985 / 0.9399 / 121.24`, which reinforces that conservative continuation
  is stable across multiple panels even though its route/exit preservation is
  slightly softer than `16045`.
  The `16081` refinement panel is also complete at
  `0.9998 / 0.9410 / 121.44`; it adds content on top of `16045` but still gives
  back a little route and exit, so it is a real stable refinement rather than a
  new stable control.
- Whether any recipe transferred to `1821` / `1842`:
  weakly at best. Cluster E is now closed and the `1821` selector carryover
  fell back to the old medium-source held-confirm regime under confirm.
- Whether the remaining ceiling looks like stability or capacity:
  current evidence says the catastrophic shortcut instability and the held-
  confirm ceiling are separable enough to map cleanly. Stability improved
  across multiple panel roots, but the held-confirm ceiling still has not
  moved.
- Single next experiment:
  start from the stable `16045` bridge and add a narrow content-focused
  auxiliary or distillation target on the factorized content branch only, while
  keeping the stabilized late-route training and selection protocol fixed.
