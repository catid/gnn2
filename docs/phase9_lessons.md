# Phase 9 Lessons

## What Phase 9 Settled

- Frozen-state content is not one story. Strong direct-entry `1874`, medium
  teacher-shaped `1842`, and the `1201` upper bound are strongly decodable.
  Fragile route-faithful `1879` is not.
- Because of that split, strict head-only shaping has to be judged by source
  family. On `1874` and `1842` the problem is reader/objective/generalization.
  On `1879` the source itself is weak-content.
- The main phase-9 answer is therefore two-part:
  - head-only shaping is genuinely viable on decodable frozen sources
  - but the remaining ceiling is held-confirm generalization, not route
    retention

## What Worked

- The frozen-state audit was the key first move. It cleanly separated
  “reader-limited” sources from “content-poor” sources and prevented another
  phase of mixing those failure modes together.
- Strong-source frozen readers materially improved content without touching
  memory. Query-gated final-query weighting, query-FILM, and content-only
  distillation all beat the older plain queryreadout family on base content
  while preserving late route.
- The strongest aggregate confirmed branch is now the strong-source
  query-gated final-query-weighted family. Query-FILM is the strongest
  alternative-reader family and is nearly tied on held-confirm content.
- Medium-source head-only recovery is also real. The `1821` family can support
  stable route-faithful head-only refinement, but it still plateaus around the
  same held-confirm content ceiling.

## What Failed Fairly

- Fragile `1879` is now a fair negative under strict head-only shaping. Across
  the multi-seed panel, route and exits stay almost perfect under stress while
  final-query accuracy stays near chance.
- Minimal-safe read-path touches on `1879` did not change that conclusion.
  They preserved route but did not produce meaningful content gain, so phase 9
  did not justify any broader reopening of `memory_`.
- Head-only ES is not the missing optimizer. In the reduced head space it can
  preserve route, but it still does not beat the better gradient-trained
  readers on content.
- Content-only distillation is real but not dominant. It produced a competitive
  strong-source family, but the lead rerun drifted enough that it should be
  treated as competitive rather than the final confirmed winner.

## What Changed Compared With Phase 8

- Phase 8 showed that teacher-free route-faithful basin entry was possible.
- Phase 9 showed that route-faithful entry is still not the right unit of
  analysis. Some route-faithful basins already contain strongly recoverable
  content; others do not.
- That means the next phase should stop treating all teacher-free basins as
  equivalent and should stop spending budget on fragile weak-content sources
  once the audit says they are weak.

## Best Current Interpretation

- On decodable strong sources, the frozen state already contains useful answer
  information.
- The current head family can read enough of it to make base content good, but
  not enough to generalize strongly under held confirmations.
- On fragile weak-content sources, head-only shaping is fundamentally starved.
- So the remaining bottleneck is not “how do we preserve route?” It is “how do
  we increase content readout capacity or read-path adaptation without letting
  route drift?”

## Single Next Experiment

The next experiment should be a single tightly constrained read-path adapter on
the strongly decodable `1874` family, starting from the best frozen-head reader
baseline and explicitly regularizing against route drift. It should not spend
more budget on fragile `1879`, new discovery tricks, or broader memory
reopening.
