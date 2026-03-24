## gnn2-e42

Question: after `14313` established a narrow local optimum for delayed-only
teacher content on the frozen-`1201` boundary, is the useful effect concentrated
in the early half, the late half, or a light overlap inside the same `24..72`
window?

Baseline references:

- `13711` selected val: `0.6567 / 0.3923 / 0.8014 / 123.76`
- `14313` selected val: `0.6626 / 0.4002 / 0.7994 / 124.49`

Metric order is `overall / fq_acc / fq_route / fq_exit`.

Window-isolation sweep at the same `lw=0.1375`:

- `14611` early half `24..48`: `0.6343 / 0.3505 / 0.8054 / 124.24`
- `14612` late half `48..72`: `0.6587 / 0.3694 / 0.8014 / 123.91`
- `14613` early overlap `24..60`: `0.6533 / 0.3545 / 0.7825 / 124.03`
- `14614` late overlap `36..72`: `0.6489 / 0.3793 / 0.7736 / 123.67`

Interpretation:

- `14612` shows the late half is materially stronger than the early half, but
  it still falls short of the full `24..72` window on final-query content.
- `14611` keeps route reasonably well but gives back far too much content and
  overall fit to matter.
- `14613` and `14614` show that simply keeping a midpoint-spanning overlap is
  not enough; both overlap windows underperform `14313` on the content-route
  balance, and both are worse than the clean late-half control on route.

Conclusion:

- the full contiguous `24..72` delayed-only window remains the local optimum in
  this timing family
- the useful effect is not reducible to a single half-window or a simple
  overlap-biased subwindow
- no rerun was justified because none of the four variants cleared `14313` or
  `13711` on the selected content-route tradeoff

So `gnn2-e42` closes as a fair local negative. The sharpened map is that both
early and late exposure appear to contribute, but only the full contiguous
window preserves the narrow `14313` tradeoff.
