# Phase 13 Lessons

This is the final phase-13 lessons document.

## Current Lessons

1. The main blocker after phase 12 really was stability, not generic reader
   capacity.
   Phase 13 already shows that conservative continuation from `15051` into the
   `15057`-style reader materially improves seed stability.

2. Stability alone has not yet moved the held-confirm ceiling.
   The best verified stability recipes still confirm near
   `full_locked fq_acc ~= 0.312-0.315` with
   `fq_route ~= 0.876-0.877` and `fq_exit ~= 115.2-115.5`.

3. Summary-time locked-route gains are not enough.
   Cluster C showed that stronger summary-time selection on hard cases can look
   real and still disappear under independent confirm.

4. The strongest current control is a bridge, not a direct regularizer.
   `16045` and the extra `16041` panel are more informative than any single
   trust-region or hard-case weighting variant because both preserve the late-
   route regime across full five-seed panels.

5. The phase-12 shortcut-collapse story is now narrower.
   The original `15057_rerun1` collapse did not cleanly reproduce on the current
   stack. Phase 13 therefore has to separate two claims:
   instability of the original direct-training path, and the separate held-
   confirm ceiling that remains even after stabilization.

6. “Stable late-route” is not one thing.
   The completed `16022` panel shows a weaker failure mode:
   one seed stayed late-route on route and exit while collapsing content badly
   on the locked slice and another recovered only partially.
   The completed `16066` panel shows a milder version of the same pattern.
   So phase 13 now has to distinguish catastrophic shortcut collapse from
   softer late-route content-collapse basins.

7. Post-stability refinement improves content more readily than route/exit.
   The completed `16081` panel stayed fully late-route and pushed content near
   saturation at `0.9998 / 0.9410 / 121.44`, but it still softened route and
   exit relative to `16045` and still did not move the held-confirm ceiling.

## Next Experiment

1. Keep the `16045` stabilization recipe fixed and add a narrow content-focused
   auxiliary or distillation target on the factorized content branch only.
   Phase 13 already showed that stability is no longer the main unknown. The
   next clean question is whether explicit content supervision can move held-
   confirm recovery without giving back late-route fidelity.
