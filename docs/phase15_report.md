# Phase 15 Report

## Status

Active. Strong-mapping read is already forming, but panel depth and Cluster F are still in progress.

## Main Question

Can a richer content-only path on top of the fixed `16045` route anchor improve held-confirm content recovery on `1874` without changing routing, control, or exit behavior?

## Current Read

Phase 15 has already separated the richer-content-path question into two cleaner sub-answers.

First, a richer path by itself is not enough. The best multi-slot baseline so far is `18026`, the `4`-slot shared-mean channel. It was rerun-clean, preserved the late-route regime under locked confirm, and remained the best multi-slot stable control. But locked confirm still fell back to the old ceiling:

- `18026` locked confirm `full_locked overall / fq_acc / fq_route / fq_exit = 0.6527 / 0.2874 / 0.8850 / 116.22`
- `18026` locked confirm `finalquery_heavy = 0.4427 / 0.2972 / 0.8789 / 115.56`

The strongest sidecar baselines closed the same way. `18035`, `18036`, `18059`, and `18060` all cleared rerun and locked confirm without route collapse, but each landed on the same confirm regime as `18026`. Interpretation: richer readout capacity alone is not sufficient; the ceiling is not broken just by replacing the single-slot path with a wider isolated content path.

Second, the best phase-15 gains so far come from richer sidecar paths paired with stronger training contracts, not from multi-slot channels. The current best dual-anchor recipe is `18057`, the `sidecarkv4 + dual-anchor + payloadaux040` branch. It is the first fully paneled headline branch in phase 15:

- `18057` five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9980 / 0.9966 / 0.9425 / 121.46`
- `18057` five-seed selected `finalquery_heavy = 0.9967 / 0.9958 / 0.9397 / 121.10`
- `18057` locked confirm `full_locked = 0.6589 / 0.2995 / 0.8797 / 116.14`

Interpretation: the richer sidecar path can buy a modest held-confirm content lift over the old phase-14 ceiling while staying in-range on route/exit, but the gain is still small. This is not a positive exit.

The current best content-only supervision branch is `18052`, the `sidecarkv4 + teacher16081 + hard-slice selector` family. Its rerun and locked confirm are already complete:

- `18052` locked confirm `full_locked = 0.6592 / 0.3001 / 0.8797 / 116.14`
- `18052` locked confirm `finalquery_heavy = 0.4515 / 0.3083 / 0.8785 / 115.97`

Its panel is still being filled, but the first two seeds are consistent with a stable late-route summary regime:

- `18152` selected `full_locked dqf = 0.9985 / 0.9384 / 121.02`
- `18153` selected `full_locked dqf = 0.9990 / 0.9421 / 121.34`

Interpretation: the richer sidecar path is currently beating the richer multi-slot path on the real bottleneck, but only by a narrow margin.

## Cluster A Read

Cluster A is mapped enough to answer its main question.

- best multi-slot branch: `18026`
- rerun: pass
- locked confirm: stable but ceiling-limited
- current interpretation: multi-slot channels do not beat the best sidecar families on held-confirm content

The other multi-slot families (`18021`, `18023`, `18027`, `18028`, `18029`, `18030`) all looked promising on summary slices but failed the confirm hard-slice gate. So the multi-slot answer is now fairly sharp: wider slot structure is not sufficient by itself.

## Cluster B Read

Cluster B is the current phase-15 winner family.

- early sidecar baselines (`18031`, `18033`, `18034`, `18035`, `18036`, `18037`) showed that a route-isolated sidecar can improve the confirm hard slice without destabilizing routing
- stronger-path sidecar follow-ups (`18059`, `18060`) proved that better hard-slice behavior alone still is not enough; both landed on the old confirm ceiling
- the current best headline branches are `18052` and `18057`

Interpretation: sidecar memory beats multi-slot channels as the better isolated content-path direction, but only modestly so far.

## Cluster C Read

Cluster C is currently more encouraging than plain richer-path scaling.

The best content-only supervision branch is `18052`. It won the confirm hard slice over `18011`, reduced late wrong-content cases, cleared the exact rerun gate, and slightly improved locked confirm content over both the old phase-14 ceiling and the richer-path baselines.

Interpretation: once the path is richer, content-only supervision still helps, but the lift remains modest.

## Cluster D Read

Cluster D is also live and competitive.

The best dual-anchor richer-path branch is `18057`. It is the first phase-15 branch to complete the full rerun + five-seed panel + locked confirm ladder, and it currently defines the cleanest fully verified phase-15 headline result.

Interpretation: dual-anchor training matters more once the content path is actually richer, but it still does not create a breakout confirmed regime.

## Cluster E Read

Characterization is not fully written yet, but one read is already clear:

- the best fully paneled phase-15 branch `18057` keeps selected full-locked exit in the same late-route band as the strong phase-14 branches
- the modest held-confirm lift therefore does not come from collapsing compute or exiting earlier under confirm

The full compute-quality frontier should compare `16045`, `18026`, `18052`, and `18057`.

## Cluster F Read

Staged, not executed yet in this report slice.

Queued sanity branches:

- `18221`, `18222` on `1821`
- `18291`, `18292` on `1879`

## Provisional Conclusion

The phase-15 map is already sharper than phase 14:

- richer content paths do help, but only the sidecar family has produced any confirm lift worth keeping
- multi-slot channels have not matched the sidecar family on the real bottleneck
- content-only supervision and dual-anchor training both help on top of the richer sidecar path
- the current lifts are modest, not ceiling-breaking

If the remaining panel roots and Cluster F do not change this, the most likely interpretation is that the next ceiling is no longer simple readout width. It is probably content writing / retrieval quality inside the isolated content path.
