# Phase 14 Cluster Scorecards

## Status

Draft in progress.

## Cluster A

Mapped enough to score as a stable negative.

- Best fully paneled branches: `17024` and `17025` teacher-first content-hidden distillation
- Hard-slice gate: pass
- Exact rerun: pass
- Locked confirm: fail on held-confirm content
- Five-seed panel: completed, stable late-route regime retained

Headline metrics for `17024`:

- selected `full_locked` confirm DQF: `0.2995 / 0.8797 / 116.14`
- selected `finalquery_heavy` confirm DQF: `0.3083 / 0.8785 / 115.97`
- five-seed mean `full_locked` summary DQF: `0.9953 / 0.9406 / 121.29`
- five-seed mean `finalquery_heavy` summary DQF: `0.9960 / 0.9429 / 121.51`

Headline metrics for `17025`:

- selected `full_locked` confirm DQF: `0.2874 / 0.8850 / 116.22`
- selected `finalquery_heavy` confirm DQF: `0.2968 / 0.8789 / 115.56`
- five-seed mean `full_locked` summary DQF: `0.9981 / 0.9427 / 121.54`
- five-seed mean `finalquery_heavy` summary DQF: `0.9968 / 0.9345 / 120.77`

Scorecard:

- Stability: strong
- Content-failure hard-slice targeting: real
- Held-confirm lift: no
- Retirement reason: even the stronger teacher-first follow-up `17025` reproduces the same locked-confirm ceiling, so this lane is now active only as background context for the hard-slice branches

## Cluster B

Mapped enough to score as a stable negative.

- Best branch: `17031` dual-anchor base
- Hard-slice gate: pass
- Exact rerun: pass
- Locked confirm: fail on held-confirm content
- Five-seed panel: completed, stable late-route regime retained

Headline metrics for `17031`:

- selected `full_locked` confirm DQF: `0.2874 / 0.8850 / 116.22`
- selected `finalquery_heavy` confirm DQF: `0.2968 / 0.8789 / 115.56`
- five-seed mean `full_locked` summary DQF: `0.9974 / 0.9432 / 121.55`
- five-seed mean `finalquery_heavy` summary DQF: `0.9955 / 0.9424 / 121.63`

Scorecard:

- Stability: strong
- Content-failure hard-slice targeting: real but modest
- Held-confirm lift: no
- Retirement reason: stable late-route branch still collapses to the old held-confirm content ceiling

## Cluster C

Still active. Current strongest no-teacher hard-slice branch `17043` is rerun-clean but confirm-ceiling-limited; teacher-backed hard-slice branches `17041` and `17042` remain the live frontier, and the first `17041` five-seed panel root is now running.

## Cluster D

Pending final scorecard.

## Cluster E

Early read already available.

- `17011`, `17024`, and `17031` all occupy a very similar summary-time late-route compute band around `full_locked exit ~= 121-123`
- quality differences on that summary-time band are small relative to the confirm collapse
- current interpretation: compute alone is not the main missing ingredient on the stabilized family

## Cluster F

Pending final scorecard.
