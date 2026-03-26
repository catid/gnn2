# Phase 15 Handoff

## Status

Active.

## Current Frontier

The cleanest current map is:

- `18026` is the best multi-slot branch. It is now rerun-clean, fully paneled, and route-stable. Five-seed selected `full_locked = 0.9978 / 0.9961 / 0.9343 / 120.87`, but locked confirm still falls back to `full_locked overall / fq_acc / fq_route / fq_exit = 0.6527 / 0.2874 / 0.8850 / 116.22`.
- `18035`, `18036`, `18059`, and `18060` show that stronger isolated sidecar branches can win the confirm hard slice and still collapse to the same locked-confirm ceiling.
- `18057` is the current best fully verified headline branch. It is rerun-clean, fully paneled, and locked-confirmed. Five-seed selected `full_locked = 0.9980 / 0.9966 / 0.9425 / 121.46`; locked confirm `full_locked = 0.6589 / 0.2995 / 0.8797 / 116.14`.
- `18052` is the best current content-only supervision branch on the richer sidecar path. It is now rerun-clean, fully paneled, and locked-confirmed. Five-seed selected `full_locked = 0.9985 / 0.9985 / 0.9397 / 121.19`; locked confirm `full_locked = 0.6592 / 0.3001 / 0.8797 / 116.14`.
- `18026` paneling is now complete through all five seeds.
- `18060` stronger sidecar paneling is now complete through all five seeds. Five-seed selected `full_locked = 0.9978 / 0.9968 / 0.9386 / 121.18`, but locked confirm still falls back to `0.6527 / 0.2874 / 0.8850 / 116.22`.
- `18221` and `18222` are live as the bounded `1821` portability sanity pair, with `18291` and `18292` still queued as the negative-control deck.

## Current Answers

- Best confirmed richer-content-path result so far: `18052` by a small margin on locked confirm content, and it is now also fully paneled.
- Multi-slot did not beat sidecar. `18026` is the best multi-slot result, and that conclusion now holds at five-seed panel depth as well as locked confirm.
- Dual-anchor training helped once the content path became richer. `18057` is stronger than the plain richer-path baselines, but not enough to count as a breakthrough.
- Content-only supervision also helped on the richer sidecar path. `18052` is slightly better than `18057` on locked confirm content, but the margin is small.
- Portability / negative-control answer: still pending the live Cluster F sanity deck.
- Current best interpretation: the remaining ceiling looks more like content writing / retrieval quality inside the isolated richer path than simple readout width or simple supervision weakness.

## Single Next Step

Finish the bounded Cluster F portability / negative-control deck so phase 15 can close with the panel-depth sidecar-vs-multislot comparison already locked and the required sanity transfer read in hand.
