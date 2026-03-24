# Phase 10 Lessons

## What Phase 10 Settled

- The phase-9 held-confirm ceiling on strong decodable sources is real. It is
  not an artifact of one frozen-head family.
- `final_sink_state` is the decisive frozen content view. Packet-only decoding
  is weak. Extra views mostly improve base fit, not held-confirm content.
- Tiny iterative readers do not unlock a better decode regime. They fit base
  well and still saturate at the same held-confirm ceiling.
- Strict downstream read-path adapters are not broadly effective. Most preserve
  route and remain flat on held confirms.
- The one attractive exception, the query-FILM low-rank adapter on strong
  source `1874`, did reproduce exactly as a single-seed outlier but failed to
  hold up across its five-seed panel. That means phase 10 should treat it as a
  useful boundary, not a confirmed new winner.

## What Worked

- Strong-source anchor reproduction was clean enough that phase 10 did not
  spend time chasing stale baselines.
- The view-ablation block was worth doing. It replaced a vague “maybe more
  views help” story with a concrete answer: `final_sink_state` is carrying the
  useful content.
- The portability block on `1842` and `1821` was also worthwhile. It showed
  that the strong-source reader ideas are not purely one-source hacks even
  though they do not solve held confirms on the secondary families.
- The seed-panel work mattered. Without it, the multiview families would have
  looked like stronger breakthroughs than they really were.

## What Failed Fairly

- Packet-only decoding is not the answer.
- Multi-view fusion is not the missing held-confirm ingredient by itself.
- Tiny iterative decoders are not the missing held-confirm ingredient by
  themselves.
- Head-level ES is not the missing optimizer once routing is frozen.
- Most route-preserving adapters are either flat or base-only improvements.

## What Changed Compared With Phase 9

- Phase 9 established that strong frozen sources are decodable and that
  stronger readers can recover substantial base content.
- Phase 10 showed that “stronger reader” is not enough as a generic strategy.
  Reader capacity alone mostly improves base behavior and leaves the same
  held-confirm ceiling.
- The bottleneck now looks even narrower than it did at the end of phase 9:
  not generic reader strength, and not generic route-safe adaptation either,
  but a very specific confirmation-aware training signal on top of the already
  best frozen readers.

## Best Current Interpretation

- The strong-source frozen state already contains answer information.
- The best one-shot readers can almost saturate base decoding, which means the
  frozen content is not the main problem on `1874`.
- Held confirms are where the real limit shows up, so the remaining failure is
  either:
  - a narrow objective or train-distribution mismatch, or
  - a need for a very small read-path touch that is better aligned than the
    current affine/residual/query-gated adapters.

## Single Next Experiment

The next experiment should be narrow:

- start from the best confirmed `1874` strong-source frozen-head baseline,
- add explicit confirmation-aware training or route-safe rollback,
- and avoid spending more budget on generic reader-family search, fragile
  `1879`, or ES.
