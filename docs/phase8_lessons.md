# Phase 8 Lessons

## What Actually Mattered

- Strong teacher quality mattered more than extra teacher channel count.
- Wait/release-only supervision was the winning teacher channel choice.
- Long release plus delayed dropout was the key schedule that turned teacher guidance into a real teacher-free basin-entry effect.
- Medium wait/release-only teachers were already enough to create route-faithful teacher-free basins, even when content stayed weak.
- The `1821 -> memoryreadout longer lowlr` branch is the best new systematic staged-recovery pipeline from a teacher-seeded source.

## What Looked Plausible But Failed

- Control-state teacher supervision did not help basin entry and usually made it worse.
- Medium full-route upper-bound imitation did not solve basin entry.
- Transfer-style weaker teachers were not enough.
- Detached warmup / late-window discovery did not fix early-exit discovery.
- Alternate controller families stayed in the same broad route-dead regime.
- REINFORCE stayed in the same immediate-exit basin.
- ES after small teacher-free recovered sources did not create a new content win.

## What Is Robust vs Fragile

- Robust:
  - teacher-seeded route-faithful basin entry on routing and exit timing
  - the strong-teacher wait/release-only recipe as a real multi-seed route phenomenon
  - the medium-teacher `1821 -> memoryreadout` recovery family
  - head-only reopening as a way to preserve fragile teacher-free route basins
- Fragile:
  - content quality after direct teacher-seeded basin entry
  - memory-based reopening on fragile teacher-free basins
  - any claim that more teacher channels automatically help

## Strongest Remaining Bottleneck

The main bottleneck is no longer “can the model learn to wait and route late at
all?” Phase 8 answered that.

The real remaining problem is:

- after teacher-free route-faithful basin entry,
- how do we improve content quality,
- without reopening `memory_` and destroying the basin?

## Next Worker: Start Here

Run a teacher-seeded wait/release-only direct-entry experiment, freeze the
memory/router/control path once the route-faithful basin is entered, and then
use only head-level content shaping:

- `readout` only
- `sink_proj + readout`
- optionally explicit content-focused auxiliary supervision

Reason:

- phase 8 says basin entry itself is now real,
- phase 8 also says `memory_` is the destabilizer on fragile teacher-free basins,
- so the next step is not “more basin entry tricks,” it is “content repair
  without touching memory.”
