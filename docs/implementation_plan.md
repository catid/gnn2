# Hard-Routing Packet GNN Experiment Plan

## Goal

Build a reproducible synthetic experiment suite that tests whether an
EGGROLL-inspired hybrid low-rank evolutionary search method makes a
hard-routing spatio-temporal packet-routing graph network practical on a
single 2-GPU machine.

## Honest Scope

This implementation will not attempt an exact reproduction of the EGGROLL paper.
Instead it will test the core practical idea under tractable constraints:

- real hard routing in the forward pass
- forward-only population evaluation for routing/control parameters
- low-rank perturbations for ES updates
- 2-GPU population sharding with `torchrun`
- direct optimization of task quality plus compute penalties

Planned simplifications:

- line-graph topology first; ring/DAG optional
- shared node core plus learned node embeddings
- synthetic offline benchmarks only
- frozen or mostly-frozen content core during hybrid ES
- low-rank adapters optional rather than mandatory

## Build Order

1. Implement synthetic benchmarks:
   - Benchmark A: mixed oracle routing
   - Benchmark B: long-horizon temporal memory
2. Implement packet-routing temporal graph model:
   - persistent node state
   - mailbox-based DELAY semantics
   - sink aggregation
   - hard and soft routing modes
3. Implement training methods:
   - soft differentiable routing baseline
   - hard straight-through baseline
   - hybrid low-rank ES for router/control + optional adapters
4. Add distributed ES execution, profiling, checkpointing, and logging.
5. Add tests, smoke scripts, main configs, and result summarization.
6. Run smoke tests and at least one meaningful comparison suite.
7. Generate `docs/experiment_report.md` with plots, tables, and a direct answer.

## Primary Comparisons

- Soft routing vs hard straight-through vs hybrid ES
- Warm-start vs no warm-start for hybrid ES
- ES perturbation rank
- Population size
- Horizon scaling on Benchmark B

## Success Conditions

The repo should end with:

- working model and datasets
- multi-GPU hybrid ES
- reproducible smoke and main commands
- saved metrics, plots, and summary tables
- an honest report stating where hybrid ES wins, loses, and whether it looks
  promising enough for further research
