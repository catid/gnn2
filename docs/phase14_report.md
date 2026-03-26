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
