# Phase 15 Candidate Packs

One pack per promoted headline branch.

Required contents for each pack:

- exact config path and checkpoint metadata
- baseline used for comparison
- content-failure hard-slice metrics
- same-seed rerun result
- five-seed panel result
- locked-confirm result
- phase15 verify checksum and output path
- final `pass` / `fail` promotion decision

Recommended filenames:

- `<candidate_name>.json`
- optional supporting CSV / JSON from `phase15_hardslice`

No branch is considered promoted until its candidate pack is complete.
