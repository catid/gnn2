# Phase 15 Cluster Scorecards

## Status

Active draft.

## Cluster A

Mapped enough to score as a stable negative.

- Best branch: `18026` multislot4 shared-mean
- Rerun: pass
- Five-seed panel: complete
- Locked confirm: pass on route/exit stability, fail on held-confirm content
- Current answer: multi-slot channels did not break the ceiling

Headline metrics for `18026`:

- five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9978 / 0.9961 / 0.9343 / 120.87`
- locked confirm `full_locked overall / fq_acc / fq_route / fq_exit = 0.6527 / 0.2874 / 0.8850 / 116.22`
- locked confirm `finalquery_heavy = 0.4427 / 0.2972 / 0.8789 / 115.56`

Scorecard:

- Stability: strong
- Hard-slice behavior: acceptable
- Held-confirm lift: no
- Retirement reason: best multi-slot branch still lands on the old ceiling after rerun, five-seed panel, and locked confirm

## Cluster B

Current winner family.

- Best clean family: sidecar memory
- Best baseline stable-sidecar map: `18035`, `18036`, `18059`, `18060`
- Best promoted sidecar branches so far: `18052`, `18057`

Scorecard:

- Stability: strong
- Route isolation: preserved
- Held-confirm lift: modest, not decisive
- Current answer: sidecar memory beats multi-slot channels as the richer-path direction, and that read now holds at panel depth

## Cluster C

Live and currently competitive.

- Best branch: `18052` sidecarkv4 + teacher16081 + hard-slice selector
- Rerun: pass
- Locked confirm: modest lift over phase-14 ceiling
- Five-seed panel: complete

Headline metrics for `18052`:

- five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9985 / 0.9985 / 0.9397 / 121.19`
- locked confirm `full_locked overall / fq_acc / fq_route / fq_exit = 0.6592 / 0.3001 / 0.8797 / 116.14`
- locked confirm `finalquery_heavy = 0.4515 / 0.3083 / 0.8785 / 115.97`

Scorecard:

- Stability: strong
- Hard-slice targeting: useful
- Held-confirm lift: small but real
- Current answer: richer-path content-only supervision helps, but it is not enough to break the ceiling

## Cluster D

Live and currently the cleanest fully verified headline lane.

- Best branch: `18057` sidecarkv4 dual-anchor payloadaux040
- Rerun: pass
- Five-seed panel: complete
- Locked confirm: modest lift over phase-14 ceiling
- Stronger baseline comparison: `18060` is now fully paneled and still confirm-ceiling limited

Headline metrics for `18057`:

- five-seed selected `full_locked overall / fq_acc / fq_route / fq_exit = 0.9980 / 0.9966 / 0.9425 / 121.46`
- locked confirm `full_locked overall / fq_acc / fq_route / fq_exit = 0.6589 / 0.2995 / 0.8797 / 116.14`

Scorecard:

- Stability: strong
- Hard-slice targeting: strong
- Held-confirm lift: modest
- Current answer: dual-anchor training matters more once the path is richer, but it still does not create a new confirmed regime

## Cluster E

Pending final writeup.

Current partial read:

- `18057` keeps selected exit in the same useful late-route compute band as the phase-14 winners
- `18052` does the same while slightly edging `18057` on locked-confirm content
- current confirm lift is not caused by an obvious compute collapse

## Cluster F

Live, not yet complete in this draft slice.

Current bounded sanity runs:

- `18221` complete on `1821`
- `18222` complete on `1821`
- `18222_rerun1` live on `1821`
- `18291` complete on `1879`
- `18292` live on `1879`
