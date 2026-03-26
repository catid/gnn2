# Phase 14 Report

## Status

Draft in progress.

## Main Question

Can narrow content-branch-only supervision on top of the stabilized `16045` route anchor improve held-confirm content recovery on `1874` without sacrificing the stable late-route regime?

## Current Read

The campaign is still verification-heavy rather than discovery-heavy.

One branch is already sharply mapped: dual-anchor route/content training has not produced a decisive phase-14 winner so far. The cleanest dual-anchor candidate remains `17031`, which improved the content-failure hard slice against the `17011` stable `16045` anchor and stayed rerun-clean, but its locked confirm still fell back to the familiar held-confirm regime:

- `17031` selected `full_locked` confirm DQF: `0.2874 / 0.8850 / 116.22`
- `17031` selected `finalquery_heavy` confirm DQF: `0.2968 / 0.8789 / 115.56`

The completed five-seed panel for the `17031` family now shows that the branch is stable in the late-route regime under fresh seeds, but stable in the wrong way for the main question:

- `17031` five-seed mean `full_locked` DQF: `0.9974 / 0.9432 / 121.55`
- `17031` five-seed mean `finalquery_heavy` DQF: `0.9955 / 0.9424 / 121.63`
- `17031` five-seed mean `longdistance` DQF: `0.9971 / 0.9508 / 153.24`

Interpretation: the dual-anchor contract preserves late-route behavior and keeps content high on summary-time slices, but it still does not convert those gains into held-confirm content recovery once the branch is independently confirmed. The main phase-14 question is therefore still centered on whether the teacher-first content-branch-only lane or the hard-slice mining lane can move that confirm ceiling.

The teacher-first lane has now cleared its first full panel root as well. `17024` kept the same summary-time pattern across five fresh seeds:

- `17024` five-seed mean `full_locked` DQF: `0.9953 / 0.9406 / 121.29`
- `17024` five-seed mean `finalquery_heavy` DQF: `0.9960 / 0.9429 / 121.51`
- `17024` five-seed mean `longdistance` DQF: `0.9961 / 0.9486 / 152.96`

But its independently confirmed held-confirm result is still stuck at the old regime:

- `17024` selected `full_locked` confirm DQF: `0.2995 / 0.8797 / 116.14`
- `17024` selected `finalquery_heavy` confirm DQF: `0.3083 / 0.8785 / 115.97`

Interpretation: content-only teacher supervision on top of the fixed `16045` route anchor clearly preserves the late-route summary regime and improves the content-failure hard slice, but the first fully paneled teacher-first candidate still does not convert that summary-time gain into held-confirm content recovery.

That same conclusion now holds for the stronger teacher-first follow-up `17025`. Its five-seed panel stayed in the same stable late-route band:

- `17025` five-seed mean `full_locked` DQF: `0.9981 / 0.9427 / 121.54`
- `17025` five-seed mean `finalquery_heavy` DQF: `0.9968 / 0.9345 / 120.77`
- `17025` five-seed mean `longdistance` DQF: `0.9963 / 0.9428 / 152.23`

But its locked confirm still fell back to the familiar ceiling:

- `17025` selected `full_locked` confirm DQF: `0.2874 / 0.8850 / 116.22`
- `17025` selected `finalquery_heavy` confirm DQF: `0.2968 / 0.8789 / 115.56`

Interpretation: simply scaling teacher-first content-branch supervision does not create a new confirmed regime. The remaining live frontier is therefore the teacher-backed hard-slice lane, not more plain teacher-first repetition.

The first completed teacher-backed hard-slice panel root `17041` also closed as a stable negative:

- `17041` five-seed mean `full_locked` DQF: `0.9948 / 0.9376 / 121.10`
- `17041` five-seed mean `finalquery_heavy` DQF: `0.9971 / 0.9396 / 121.28`
- `17041` locked confirm `full_locked` DQF: `0.2874 / 0.8850 / 116.22`

Interpretation: teacher-backed hard-slice mining preserves the stable late-route regime and survives rerun/panel stress, but it still collapses onto the same held-confirm ceiling under locked confirmation. The remaining live frontier is now the stronger weighted teacher-backed branch `17042` plus the `17043` no-teacher panel control.

## Cluster E Early Read

The first compute-quality frontier check is already informative even before the remaining panel roots complete. On summary-time slices, the stable baseline `17011`, the first teacher-first branch `17024`, and the first dual-anchor branch `17031` all sit on nearly the same late-route compute band:

- `17011` selected `full_locked` DQF: `0.9972 / 0.9402 / 121.33`
- `17024` selected `full_locked` DQF: `0.9981 / 0.9346 / 120.62`
- `17031` selected `full_locked` DQF: `0.9972 / 0.9570 / 123.08`

Interpretation: variable thinking time is not the missing axis by itself on the current stable family. The current branches already buy roughly the same late-route compute, and the main separation remains whether any branch can turn that stable compute regime into held-confirm content recovery rather than just producing another summary/confirm split.

## Cluster D Read

The gated content-only sidecar branch is already mapped enough to narrow the next step. The best sidecar candidate was `17063`, which stayed in the same summary-time late-route regime:

- `17063` selected `full_locked` summary DQF: `0.9991 / 0.9533 / 122.61`
- `17063` selected `finalquery_heavy` summary DQF: `0.9988 / 0.9363 / 120.90`

It also cleared the hard-slice gate and exact rerun. But its locked confirm still fell back to the same held-confirm regime:

- `17063` selected `full_locked` confirm DQF: `0.2874 / 0.8850 / 116.22`
- `17063` selected `finalquery_heavy` confirm DQF: `0.2976 / 0.8789 / 115.56`

Interpretation: a small content-only sidecar path can preserve the stabilized late-route regime, but the first credible sidecar branch still looks like another summary-time improvement that does not survive held-confirm verification.
