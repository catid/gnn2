# Phase 7 Lessons

## What Actually Mattered

- Route-faithful source checkpoints mattered more than almost any local loss tweak.
- Adapter ES is the most reliable way we found to preserve delayed-route fidelity once a source checkpoint already contains real late-route structure.
- Medium-source recovery is real. The `forceoracle -> memoryreadout` branch is not a one-seed fluke.
- Strong-source transfer is also real, but its failure mode is now obvious: routing transfers much more easily than content quality.
- The partial-init family showed that `sink_proj` is a stronger second-stage repair lever than `memory_` when the source is only readout-level stable.

## What Looked Plausible But Failed

- More keepalive shaping alone did not widen the direct hard-ST discovery basin.
- REINFORCE did not rescue discovery.
- Setclear / monotone / waitstate controller families stayed in the same broad early-exit regime.
- Wait-loss and related supervision knobs were not the missing ingredient.
- Router-only ES is not universally good. It collapses on the keepalive anchor and overfits the medium-source branch on locked confirmations.

## What Is Robust vs Fragile

- Robust:
  - weak-basin ES rescue followed by gradient recovery
  - keepalive-anchor adapter ES
  - the source-matched adapter-vs-router-only ES tradeoff on medium and teacher-shaped sources
  - route-transfer behavior in the partial-init cross-polish branch under locked confirmations
- Fragile:
  - direct keepalive discovery from scratch
  - content transfer after strong-source route transfer
  - any claim that a new controller family fixes discovery without a seed panel

## Strongest Remaining Bottleneck

The main unresolved bottleneck is still direct basin entry. Phase 7 made it much
clearer that:

- the benchmark is learnable,
- late-route structure can be preserved and improved once it exists,
- ES can strongly help after basin entry,
- but from-scratch hard-ST discovery still does not enter that basin reliably.

## Next Worker: Start Here

Run a teacher-seeded direct keepalive experiment that uses the verified
route-faithful adapter-ES checkpoint as a fixed controller teacher for only the
wait / release / control-state channels, then release into task-only training.

Reason:

- plain teacher distillation created useful medium sources,
- plain keepalive tuning did not widen the direct basin,
- the missing step is a route-faithful direct basin-entry recipe, not another
  generic content or controller tweak.
