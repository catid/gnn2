# Phase 10 Handoff

- Best confirmed strong-source read-path result:
  the strict frozen strong-source family still defines the robust ceiling.
  The strongest phase-10 multiview query-gated family reached five-seed means
  of base `0.9891 / 0.9775 / 0.9445 / 121.99` and full_locked
  `0.6589 / 0.2996 / 0.8797 / 116.14` for
  `overall / fq_acc / fq_route / fq_exit`.
- Best confirmed medium-source read-path result:
  the `1821` multiview query-gated fq5 family reached five-seed means of
  base `0.8920 / 0.8027 / 0.9258 / 122.78` and full_locked
  `0.6135 / 0.2557 / 0.8289 / 115.23`.
- Whether multi-view beat single-view:
  multi-view beat single-view on base fit, but not on the held-confirm ceiling.
  `final_sink_state` alone was enough to reach roughly the same locked-confirm
  regime as the stronger multiview readers.
- Whether a read-path adapter helped beyond strict heads:
  not robustly. The query-FILM low-rank adapter `10024` reproduced exactly as
  a single-seed outlier but failed across its five-seed panel.
- Whether minimal-safe partial unfreezing was needed:
  no. Phase 10 stayed disciplined and did not justify entering broader touches
  beyond the strict read path.
- Whether ES helped at the read-path level:
  no. Head-level ES was a fair negative once routing was frozen.
- Which clusters were retired and why:
  cluster B adapters were a fair negative overall because the best outlier did
  not survive panel verification; cluster E iterative readers were a fair
  negative because they hit the same held-confirm ceiling; cluster G head-level
  ES was a fair negative because it never beat the stronger gradient readers.
- Single next experiment:
  a confirmation-aware objective on the best frozen `1874` baseline, rather
  than another broad reader-family or adapter sweep.
