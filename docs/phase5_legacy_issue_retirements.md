## Legacy Phase-5 Issue Retirements

This note closes out three old phase-5 beads whose scientific questions were
answered or superseded by later verified campaigns.

### gnn2-hzo

Original question: can direct hard-ST controller discovery on Benchmark B v2 be
stabilized across seeds by better wait-controller structure, stronger temporal
consistency, or direct route-head restructuring?

What later work established:

- [phase5_report.md](/home/catid/gnn2/docs/phase5_report.md) already showed that
  the direct release-gate family failed and that hard-ST discovery was still
  not robust.
- [phase7_report.md](/home/catid/gnn2/docs/phase7_report.md) sharpened that
  result: from-scratch hard-ST discovery was still not robust across seeds even
  after broader controller-family exploration.
- [phase8_report.md](/home/catid/gnn2/docs/phase8_report.md) then replaced the
  old direct-stabilization question with a stronger one: teacher-guided
  wait/release-only supervision can create route-faithful teacher-free basin
  entry, while direct discovery remained the wrong bottleneck.

Conclusion:

- retire `gnn2-hzo` as superseded by the later basin-entry map
- the direct hard-ST stabilization program did not produce a robust win, and
  the research line moved to teacher-guided basin entry and content recovery

### gnn2-92b

Original question: is resume-based ES polish from working final-query
checkpoints a reliable refinement stage, and how does it compare with
gradient-only continuation?

What later work established:

- [phase5_report.md](/home/catid/gnn2/docs/phase5_report.md) confirmed the
  medium-basin ES rescue from `seed950` with an exact rerun, and also mapped
  the weak-basin `seed947` rescue.
- [phase7_report.md](/home/catid/gnn2/docs/phase7_report.md) expanded that into
  a broader ES role map across strong, medium, weak, and teacher-shaped
  sources: ES is highly useful after the right basin is found, but the best
  ES mode is source-dependent.
- [phase8_report.md](/home/catid/gnn2/docs/phase8_report.md) and
  [phase9_report.md](/home/catid/gnn2/docs/phase9_report.md) preserved that
  conclusion: ES remained downstream of basin entry, not the main discovery
  mechanism.

Conclusion:

- retire `gnn2-92b` as answered and superseded by a stronger later ES map
- the scientific question is no longer whether ES polish can work, but when and
  on which source families it is worth using

### gnn2-mjd

Original question: after ES repairs routing on a weak basin, can a short
gradient-only refinement recover the missing content?

What later work established:

- [phase5_report.md](/home/catid/gnn2/docs/phase5_report.md) explicitly named
  this as the next experiment after the `seed947` weak-basin rescue.
- [phase7_report.md](/home/catid/gnn2/docs/phase7_report.md) shows that this
  experiment succeeded as the weak-basin staged-recovery anchor
  [seed973_rerun1](/home/catid/gnn2/results/phase7_anchor/hard_st_b_v2_weak_es_content_refine_sinkcore_seed973_rerun1)
  with verified base metrics `0.9173 / 0.8643 / 0.8155 / 115.62`.

Conclusion:

- close `gnn2-mjd` as completed by the phase-7 weak-basin staged-recovery
  anchor
