# Phase 11 `gnn2-t2d` Notes

## Scope

Audit the exact phase-10 `1201` anchor directly, instead of judging it through
failed reader ports. The question was whether the `qiz` collapse came from
missing answer information under held confirmations, or from targeting the
wrong frozen interface.

## Audit Result

Across `base_test`, `full_locked`, and `finalqueryheavy`, the answer is the
same:

- best accuracy: `1.0000`
- best representation: `memory_read_state_query`
- best probe: `mlp_query_conditioned`
- `head_only_go_signal: true`
- decisive view accuracies:
  - `sink_state_query`: `1.0000`
  - `baseline_readout_input`: `1.0000`
  - `final_readout_input`: `1.0000`
  - `packet_state_query`: about `0.24`

## Conclusion

The exact `1201` frozen source is perfectly decodable at the sink/readout views
that matter. So `qiz` did not fail because `1201` lost content under held
confirms. It failed because the `1874`-style frozen reader geometry does not
port cleanly to `1201`.

That sharply narrowed the next follow-up:

- keep the strong `1201` core frozen
- stop inheriting the `1874` reader interface
- try a compatible zero-init or source-native head instead

That follow-up became `gnn2-rnd`.
