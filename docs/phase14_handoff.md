# Phase 14 Handoff

## Status

Complete.

## Current Frontier

The cleanest locked-in map at handoff time is:

- Dual-anchor training is not the phase-14 winner.
- `17031` is rerun-clean, hard-slice-positive, and now fully paneled, but its locked confirm still falls back to `full_locked dqf 0.2874 / 0.8850 / 116.22`.
- `17024` is now also fully paneled: five-seed `full_locked dqf 0.9953 / 0.9406 / 121.29`, but confirm still falls back to `full_locked dqf 0.2995 / 0.8797 / 116.14`.
- `17025` is now fully paneled as well: five-seed `full_locked dqf 0.9981 / 0.9427 / 121.54`, but confirm still falls back to `full_locked dqf 0.2874 / 0.8850 / 116.22`.
- `17041` is now fully paneled: five-seed `full_locked dqf 0.9948 / 0.9376 / 121.10`, but confirm still falls back to `full_locked dqf 0.2874 / 0.8850 / 116.22`.
- `17043` is now fully paneled as the no-teacher hard-slice control: five-seed `full_locked dqf 0.9991 / 0.9425 / 121.70`, but confirm still falls back to `full_locked dqf 0.2874 / 0.8850 / 116.22`.
- `17063` is the best gated content-only sidecar check; it stayed rerun-clean and late-route on summary slices, but confirm still fell back to `full_locked dqf 0.2874 / 0.8850 / 116.22`.
- `17042` was the last weighted teacher-backed hard-slice follow-up worth checking beyond the paneled roots. It was rerun-clean and hard-slice-positive, but its locked confirm still fell back to `full_locked dqf 0.2874 / 0.8850 / 116.22`, so it was retired without extra panel budget.
- `17101` closes Cluster F as the strongest bounded portability sanity on `1821`: its rerun preserves the summary-time late-route regime at `full_locked dqf 0.9925 / 0.9300 / 122.67`, but locked confirm falls back to `full_locked overall/dqf 0.6136 / 0.2533 / 0.8329 / 116.34`.
- `17102` was weaker than `17101` on the decisive `1821` summary-time slice and was retired without rerun or confirm budget.
- `17103` and `17104` both stayed properly negative on `1879`, so the late-route selectors do not manufacture false-positive content recovery on the bad source.

## Final Answers

- Best confirmed content-branch-only result: `17024`, with five-seed `full_locked dqf 0.9953 / 0.9406 / 121.29`, but locked confirm still falls back to `0.2995 / 0.8797 / 116.14`.
- Dual-anchor route/content training helped summary-time stability and hard-slice behavior, but `17031` did not beat the teacher-first lane under locked confirm.
- Hard-slice mining mattered for summary-time content-failure behavior, but `17041`, `17042`, and `17043` still collapsed to the same confirm ceiling.
- A content-only sidecar path was not enough: `17063` stayed late-route and rerun-clean, but locked confirm still fell back to the same regime.
- Weak portability exists only as bounded sanity: `17101` carries over on `1821` summary-time slices, but confirm returns to the old medium-source regime; `1879` remains a clean negative control.
- The remaining ceiling now looks more like a deeper content-path architectural bottleneck than a pure content-supervision problem on the current fixed branch.
- Single next experiment: keep the `16045` route anchor fixed and replace the current single-slot content path with a slightly richer content-only decoder or multi-slot content channel that cannot affect routing, then train it directly against the phase-14 content-failure hard slice.
