# Phase 11 `gnn2-udf` Notes

## Scope

`aj5` ended with a rerun-backed local optimum on frozen `1201`:

- `12423` improved on `12412` in overall behavior and route on the base test
- but it was still unclear whether that local gain actually moved the
  held-confirm boundary

This follow-up did not add new training runs. It promoted the comparison into
the existing phase-11 verify path:

- run `phase11_verify` on `12423`
- run the same confirm suite on `12412`
- compare `full_locked`, `finalquery_heavy`, and `longdistance`

## Confirm Comparison

Base test:

- `12412`: `0.6283 / 0.3429 / 0.7386 / 124.64`
- `12423`: `0.6423 / 0.3282 / 0.7848 / 123.68`

Held confirms:

- `12412 full_locked`: `0.1294 / 0.2515 / 0.9876 / 126.43`
- `12423 full_locked`: `0.1294 / 0.2515 / 0.9876 / 126.43`
- `12412 finalquery_heavy`: `0.1970 / 0.2453 / 0.9860 / 126.44`
- `12423 finalquery_heavy`: `0.1970 / 0.2453 / 0.9860 / 126.44`
- `12412 longdistance`: `0.1848 / 0.2639 / 0.9843 / 157.96`
- `12423 longdistance`: `0.1848 / 0.2639 / 0.9843 / 157.96`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

## Interpretation

- the interpolation refinement is real on the base test:
  `12423` improves overall behavior and route over `12412`
- but under the current held-confirm suite, that gain vanishes:
  `12423` and `12412` produce effectively identical confirm metrics
- so interpolation weight tuning changes the base tradeoff, but not the current
  held-confirm ceiling

## Conclusion

`gnn2-udf` is closed as a useful mapping result.

The new local optimum `12423` is still the right frozen-`1201` base boundary,
but confirm-style evaluation says it does not yet move the held-confirm
frontier. The next step is therefore not more interpolation sweeps. It is a
confirmation-aware objective or selector change starting from `12423`.
