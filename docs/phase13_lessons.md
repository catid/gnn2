# Phase 13 Lessons

This is a live phase-13 lessons document. It captures conclusions that are
already stable enough to guide the rest of the campaign.

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
   `16045` is more informative than any single trust-region or hard-case
   weighting variant because it preserves the late-route regime across a full
   five-seed panel.

5. The phase-12 shortcut-collapse story is now narrower.
   The original `15057_rerun1` collapse did not cleanly reproduce on the current
   stack. Phase 13 therefore has to separate two claims:
   instability of the original direct-training path, and the separate held-
   confirm ceiling that remains even after stabilization.

## Open Questions

1. Can the live low-LR post-stability refinements (`16081`-`16084`) move
   held-confirm content once the basin is already stable?
2. Are the best stability recipes at least weakly portable to `1821`, or do
   they immediately degrade into earlier-exit regimes?
3. Does the remaining ceiling now look fundamentally like content capacity, or
   is there still a narrower optimization issue inside the stabilized basin?
