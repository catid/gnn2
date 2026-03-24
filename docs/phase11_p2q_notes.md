## gnn2-p2q

Question: after `14313` established a narrow local optimum at `lw=0.1375`
on the same `24..72 delayed_only` window, is the remaining damage coming from
teacher pressure that stays too strong late in that window? If so, decaying the
same signal inside `24..72` should keep the route gain while preserving more
final-query content.

Baseline references:

- `14213` selected val: `0.6699 / 0.3942 / 0.7954 / 124.21`
- `14313` selected val: `0.6626 / 0.4002 / 0.7994 / 124.49`
- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Decay sweep inside the same `24..72` window:

- `14511` `lw=0.1375`, scale `1.0 -> 0.5`: `0.6553 / 0.3724 / 0.7994 / 124.29`
- `14512` `lw=0.1375`, scale `1.0 -> 0.25`: `0.6460 / 0.3545 / 0.8073 / 123.68`
- `14513` `lw=0.15`, scale `1.0 -> 0.5`: `0.6460 / 0.3625 / 0.7895 / 124.75`

Interpretation:

- `14511` essentially kept `14313` route but gave back too much final-query
  content to stay competitive.
- `14512` bought a little extra route, but the content loss was too large and
  overall fit also regressed.
- `14513` failed to retain the `14213` content lift and also fell below `14313`
  on route.

Conclusion:

- decaying the delayed-only teacher scale inside the same `24..72` window is a
  fair local negative
- none of the three decayed schedules beat either `14313` or `13711` on the
  selected content-route tradeoff
- no rerun was justified because no variant cleared the existing local
  boundaries

So `gnn2-p2q` closes as a fair local negative. The remaining narrow follow-up is
to isolate which half of the `24..72` window is doing the useful work, rather
than weakening the entire window uniformly.
