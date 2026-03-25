# Phase 12 Lessons

## What Phase 12 Settled

- Route-trace conditioning is not the missing ingredient. The best temporal-bank
  readers were route-blind or route-neutral.
- Temporal-bank readers and factorized content/query readers can drive base
  decoding on `1874` to near saturation, but they still land on the old
  held-confirm ceiling.
- Narrow payload/query auxiliary supervision improves fit but does not change
  the held-confirm regime.
- The strongest `1874` family is not just ceiling-limited; it is also unstable
  under exact rerun and can fall into an early-exit shortcut basin.
- Portability to `1821` and `1842` is weaker than the best single seeds
  suggested. Under panels, both settle into a lower-route earlier-exit regime.
- Minimal keyed sinks are a real architectural lever, but their first stable
  effect is the wrong one: an off-regime early-exit shortcut.

## What Worked

- The anchor block was worth doing. It prevented phase 12 from chasing stale
  baselines.
- The direct route-conditioned vs route-blind controls were worth doing. They
  eliminated an attractive but wrong story quickly.
- Factorized readers were worth building. Even though they did not solve held
  confirms, they clarified that the remaining problem is not generic lack of
  reader expressivity.
- The stress block mattered as much as the model block. Without the rerun and
  portability panels, `15057` would have looked like a much stronger scientific
  result than it really is.

## What Failed Fairly

- Route-trace-conditioned temporal readers did not beat route-blind controls.
- Contiguous windows did not generalize into a broad temporal-slab advantage.
- Probe-guided adapters were a serious negative even with fair rerun closure.
- Objective-side source-distill tuning on the best reader stayed in the wrong
  early-exit regime.
- Keyed sinks did not rescue held confirms. They produced a stable shortcut
  instead.

## What Changed Compared With Phase 10 and Phase 11

- Phase 10 and phase 11 already showed that better endpoint readers and narrow
  objective/output changes did not solve held confirms.
- Phase 12 shows the problem is not simply that the reader needs more temporal
  context or a cleaner content/query split.
- The map is now sharper: the strongest route-blind temporal/factorized readers
  can already read almost everything they are going to read in the easy regime.
- The remaining frontier is now dominated by stability of the late-route
  solution versus drift into a shortcut basin.

## Best Current Interpretation

- The decisive missing held-confirm gain is not sitting in an ignored route
  trace summary.
- It is not unlocked by generic temporal-bank capacity.
- It is not unlocked by factorizing content and query alone.
- It is not unlocked by a first keyed-sink extension.

The best current interpretation is:

- there is a genuine good late-route solution on `1874`,
- but the training dynamics do not preserve it robustly,
- and several seemingly stronger variants drift into a high-accuracy
  early-exit shortcut that is useless for held confirms.

## Single Next Experiment

The next experiment should be a narrow stability experiment on the best
route-blind temporal/factorized `1874` family:

- preserve the late-route solution explicitly,
- regularize against the early-exit shortcut basin,
- and avoid broad new reader families, broad sink work, or route-side changes.
