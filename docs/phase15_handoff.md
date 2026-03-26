# Phase 15 Handoff

## Status

Active.

## Current Frontier

The cleanest current map is:

- `18026` is the best multi-slot branch. It is rerun-clean and route-stable, but locked confirm falls back to `full_locked overall / fq_acc / fq_route / fq_exit = 0.6527 / 0.2874 / 0.8850 / 116.22`.
- `18035`, `18036`, `18059`, and `18060` show that stronger isolated sidecar branches can win the confirm hard slice and still collapse to the same locked-confirm ceiling.
- `18057` is the current best fully verified headline branch. It is rerun-clean, fully paneled, and locked-confirmed. Five-seed selected `full_locked = 0.9980 / 0.9966 / 0.9425 / 121.46`; locked confirm `full_locked = 0.6589 / 0.2995 / 0.8797 / 116.14`.
- `18052` is the best current content-only supervision branch on the richer sidecar path. It is rerun-clean and locked-confirmed at `full_locked = 0.6592 / 0.3001 / 0.8797 / 116.14`. Its panel is still filling.
- `18221`, `18222`, `18291`, and `18292` are queued as the bounded Cluster F portability / negative-control sanity deck.

## Current Answers

- Best confirmed richer-content-path result so far: `18052` by a small margin on locked confirm content, though `18057` is the cleaner fully paneled branch.
- Multi-slot did not beat sidecar. `18026` is the best multi-slot result, and it still fell back to the old confirm ceiling.
- Dual-anchor training helped once the content path became richer. `18057` is stronger than the plain richer-path baselines, but not enough to count as a breakthrough.
- Content-only supervision also helped on the richer sidecar path. `18052` is slightly better than `18057` on locked confirm content so far, but the margin is small and its panel is still in progress.
- Portability / negative-control answer: still pending the queued Cluster F sanity deck.
- Current best interpretation: the remaining ceiling looks more like content writing / retrieval quality inside the isolated richer path than simple readout width or simple supervision weakness.

## Single Next Step

Finish the `18052` panel, then complete one multislot panel root and one stronger-path sidecar panel root so the sidecar-vs-multislot comparison is locked in at panel depth before spending more budget on portability.
