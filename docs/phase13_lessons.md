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

1. Do the live `16022` and `16066` panels stay clean across all five seeds, or
   does any weaker late-route instability reappear once the panel coverage goes
   beyond the already completed `16045` bridge family?
2. Once the current verification floor is complete, does the remaining ceiling
   look fundamentally like content capacity rather than stability?
3. Are the queued `16041` and `16081` reruns / panels enough to close phase 13
   cleanly, or does one more stable `1874` family need full verification to
   make the final map defensible?
