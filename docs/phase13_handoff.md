# Phase 13 Handoff

This is a live handoff note and will be finalized when phase 13 closes.

- Best current confirmed stable `1874` result:
  [16045](/home/catid/gnn2/results/phase13_dev/hard_st_b_v2_teacher1874_factorized_temporalbank_query_bilinear_bridge_blend85_15051readout_15057extras_selectlocked_seed16045_p1),
  with confirm base `0.9985 / 0.9961 / 0.9470 / 122.21` and confirm
  full_locked `0.6483 / 0.3135 / 0.8771 / 115.49` for
  `overall / fq_acc / fq_route / fq_exit`.
- Whether `15057`-style upside can be stabilized:
  partly yes. Conservative continuation from `15051` into the `15057`-style
  reader survives a five-seed panel without recreating the old shortcut
  collapse.
- Whether continuation from `15051` helped:
  yes. It is the clearest positive result of phase 13 so far.
- Whether hard-case mining or held-confirm-aware selection helped:
  only at summary time. Those gains have not survived independent confirm.
- Whether any recipe transferred to `1821` / `1842`:
  not tested yet in phase 13; the first bounded secondary-source sanity runs
  are queued as Cluster E.
- Whether the remaining ceiling looks like stability or capacity:
  current evidence says the catastrophic shortcut instability and the held-
  confirm ceiling are partly separable. Stability improved, but the ceiling has
  not moved yet.
- Single next experiment:
  finish the low-LR post-stability refinement deck on `16045`, then port the
  best stable selector recipe to `1821` and `1879` as the bounded sanity check.
