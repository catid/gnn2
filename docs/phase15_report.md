# Phase 15 Report

## Status

Active. Strong-mapping read is already forming, but panel depth and Cluster F are still in progress.

## Main Question

Can a richer content-only path on top of the fixed `16045` route anchor improve held-confirm content recovery on `1874` without changing routing, control, or exit behavior?

## Current Read

Phase 15 has already separated the richer-content-path question into two cleaner sub-answers.

First, a richer path by itself is not enough. The best multi-slot baseline is still `18026`, the `4`-slot shared-mean channel. It is now rerun-clean, fully paneled, and locked-confirmed. Its five-seed selected summary regime stayed stable:

- `18026` five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9978 / 0.9961 / 0.9343 / 120.87`
- `18026` five-seed selected `finalquery_heavy = 0.9978 / 0.9973 / 0.9388 / 121.07`

But locked confirm still fell back to the old ceiling:

- `18026` locked confirm `full_locked overall / fq_acc / fq_route / fq_exit = 0.6527 / 0.2874 / 0.8850 / 116.22`
- `18026` locked confirm `finalquery_heavy = 0.4427 / 0.2972 / 0.8789 / 115.56`

The strongest sidecar baselines closed the same way. `18035`, `18036`, `18059`, and now the fully paneled `18060` all cleared rerun and locked confirm without route collapse, but each landed on the same confirm regime as `18026`. `18060` five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9978 / 0.9968 / 0.9386 / 121.18`, then locked confirm fell back to `0.6527 / 0.2874 / 0.8850 / 116.22`. Interpretation: richer readout capacity alone is not sufficient; the ceiling is not broken just by replacing the single-slot path with a wider isolated content path.

Second, the best phase-15 gains so far come from richer sidecar paths paired with stronger training contracts, not from multi-slot channels. The current best dual-anchor recipe is `18057`, the `sidecarkv4 + dual-anchor + payloadaux040` branch. It is the first fully paneled headline branch in phase 15:

- `18057` five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9980 / 0.9966 / 0.9425 / 121.46`
- `18057` five-seed selected `finalquery_heavy = 0.9967 / 0.9958 / 0.9397 / 121.10`
- `18057` locked confirm `full_locked = 0.6589 / 0.2995 / 0.8797 / 116.14`

Interpretation: the richer sidecar path can buy a modest held-confirm content lift over the old phase-14 ceiling while staying in-range on route/exit, but the gain is still small. This is not a positive exit.

The current best content-only supervision branch is `18052`, the `sidecarkv4 + teacher16081 + hard-slice selector` family. It is now rerun-clean, fully paneled, and locked-confirmed:

- `18052` five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9985 / 0.9985 / 0.9397 / 121.19`
- `18052` five-seed selected `finalquery_heavy = 0.9977 / 0.9971 / 0.9426 / 121.51`
- `18052` locked confirm `full_locked = 0.6592 / 0.3001 / 0.8797 / 116.14`
- `18052` locked confirm `finalquery_heavy = 0.4515 / 0.3083 / 0.8785 / 115.97`

Interpretation: the richer sidecar path is currently beating the richer multi-slot path on the real bottleneck, and that read now holds at panel depth rather than just single-run depth.

## Cluster A Read

Cluster A is mapped enough to answer its main question.

- best multi-slot branch: `18026`
- rerun: pass
- five-seed panel: complete
- locked confirm: stable but ceiling-limited
- current interpretation: multi-slot channels do not beat the best sidecar families on held-confirm content

The other multi-slot families (`18021`, `18023`, `18027`, `18028`, `18029`, `18030`) all looked promising on summary slices but failed the confirm hard-slice gate. So the multi-slot answer is now fairly sharp: wider slot structure is not sufficient by itself.

## Cluster B Read

Cluster B is the current phase-15 winner family.

- early sidecar baselines (`18031`, `18033`, `18034`, `18035`, `18036`, `18037`) showed that a route-isolated sidecar can improve the confirm hard slice without destabilizing routing
- stronger-path sidecar follow-ups (`18059`, `18060`) proved that better hard-slice behavior alone still is not enough; both landed on the old confirm ceiling, and `18060` now confirms that at five-seed panel depth
- the current best headline branches are `18052` and `18057`

Interpretation: sidecar memory beats multi-slot channels as the better isolated content-path direction, and that comparison is now locked in at panel depth for the multi-slot side.

## Cluster C Read

Cluster C is currently more encouraging than plain richer-path scaling.

The best content-only supervision branch is `18052`. It won the confirm hard slice over `18011`, reduced late wrong-content cases, cleared the exact rerun gate, and slightly improved locked confirm content over both the old phase-14 ceiling and the richer-path baselines.

Interpretation: once the path is richer, content-only supervision still helps, and `18052` is now the best fully paneled content-only richer-path branch, but the lift remains modest.

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

Live, but not complete yet in this report slice.

Current bounded sanity branches:

- `18221` complete on `1821`
- `18222` complete on `1821`
- `18222_rerun1` live on `1821`
- `18291` complete on `1879`
- `18292` complete on `1879`

## Provisional Conclusion

The phase-15 map is already sharper than phase 14:

- richer content paths do help, but only the sidecar family has produced any confirm lift worth keeping
- multi-slot channels have not matched the sidecar family on the real bottleneck
- content-only supervision and dual-anchor training both help on top of the richer sidecar path
- the current lifts are modest, not ceiling-breaking

If the remaining panel roots and Cluster F do not change this, the most likely interpretation is that the next ceiling is no longer simple readout width. It is probably content writing / retrieval quality inside the isolated content path.

The campaign is still filling the portability and sanity roots that matter most for the final closeout:

- `18026` multislot panel is complete through all five seeds
- `18060` stronger sidecar dual-anchor panel is complete through all five seeds
- `18222` is now the stronger summary-time `1821` portability read and `18222_rerun1` is live to decide whether it deserves any confirm budget
- `18291` and `18292` are now both closed as clean bounded `1879` negatives
