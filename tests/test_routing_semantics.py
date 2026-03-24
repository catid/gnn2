from __future__ import annotations

import torch
import torch.nn as nn

from src.models.packet_routing import CosineReadoutHead, MixtureReadoutHead, PacketRoutingModel, compute_task_classification_loss


class StubCore(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        packet_state: torch.Tensor,
        node_state: torch.Tensor,
        observations: torch.Tensor,
        node_index: torch.Tensor,
        age_fraction: torch.Tensor,
        time_fraction: torch.Tensor,
        remaining_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del node_index, age_fraction, time_fraction, remaining_fraction
        logits = observations[..., :3]
        packet_next = packet_state + observations[..., 3 : 3 + self.hidden_dim]
        return node_state, packet_next, logits


def make_model(num_nodes: int, max_internal_steps: int) -> PacketRoutingModel:
    model = PacketRoutingModel(
        {
            "num_nodes": num_nodes,
            "obs_dim": 16,
            "hidden_dim": 8,
            "num_classes": 2,
            "max_internal_steps": max_internal_steps,
            "max_total_steps": 32,
            "adapter_rank": 0,
        }
    )
    model.core = StubCore(hidden_dim=8)
    with torch.no_grad():
        model.sink_proj.weight.zero_()
        model.sink_proj.bias.zero_()
        model.readout[1].weight.zero_()
        model.readout[1].bias.zero_()
    return model


def test_hard_forward_reaches_sink_same_timestep() -> None:
    model = make_model(num_nodes=3, max_internal_steps=3)
    observations = torch.zeros(1, 1, 3, 16)
    observations[..., 0] = 5.0
    labels = torch.zeros(1, dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
    )

    assert output.stats["hops"].tolist() == [3.0]
    assert output.stats["delays"].tolist() == [0.0]
    assert output.stats["ttl_fail"].tolist() == [0.0]
    assert output.stats["exit_time"].tolist() == [0.0]
    assert output.stats["early_exit_mass"].tolist() == [1.0]


def test_hard_exit_exits_immediately() -> None:
    model = make_model(num_nodes=3, max_internal_steps=3)
    observations = torch.zeros(1, 1, 3, 16)
    observations[..., 1] = 5.0
    labels = torch.zeros(1, dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
    )

    assert output.stats["hops"].tolist() == [0.0]
    assert output.stats["delays"].tolist() == [0.0]
    assert output.stats["ttl_fail"].tolist() == [0.0]
    assert output.stats["exit_time"].tolist() == [0.0]
    assert output.stats["early_exit_mass"].tolist() == [1.0]


def test_delay_mailbox_reappears_next_external_step() -> None:
    model = make_model(num_nodes=2, max_internal_steps=1)
    observations = torch.zeros(1, 2, 2, 16)
    observations[:, 0, :, 2] = 5.0
    observations[:, 1, :, 1] = 5.0
    labels = torch.zeros(1, dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
    )

    assert output.stats["hops"].tolist() == [0.0]
    assert output.stats["delays"].tolist() == [1.0]
    assert output.stats["ttl_fail"].tolist() == [0.0]
    assert output.stats["exit_time"].tolist() == [1.0]
    assert output.stats["early_exit_mass"].tolist() == [0.0]


def test_return_trace_includes_sink_state_views() -> None:
    model = make_model(num_nodes=2, max_internal_steps=1)
    observations = torch.zeros(1, 2, 2, 16)
    observations[:, 0, :, 2] = 5.0
    observations[:, 1, :, 1] = 5.0
    labels = torch.zeros(1, dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
        return_trace=True,
    )

    assert output.trace is not None
    assert output.trace["sink_state"].shape == (1, 2, 8)
    assert output.trace["final_sink_state"].shape == (1, 8)
    assert output.trace["final_readout_input"].shape == (1, 8)
    assert torch.allclose(output.trace["final_sink_state"], output.sink_state)


def test_query_conditioned_readout_constructs_and_runs() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 3,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "readout_mode": "query_conditioned",
        }
    )
    observations = torch.zeros(2, 3, 2, 8)
    observations[:, -1, 0, 1] = 1.0
    labels = torch.zeros(2, dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
        return_trace=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.sink_state.shape == (2, 4)
    assert output.trace is not None
    assert output.trace["final_sink_state"].shape == (2, 4)
    assert output.trace["final_readout_input"].shape == (2, 8)


def test_query_gated_and_film_readouts_construct_and_run() -> None:
    for mode in ("query_gated", "query_film"):
        model = PacketRoutingModel(
            {
                "num_nodes": 2,
                "obs_dim": 8,
                "hidden_dim": 4,
                "num_classes": 3,
                "max_internal_steps": 1,
                "max_total_steps": 8,
                "adapter_rank": 0,
                "readout_mode": mode,
            }
        )
        observations = torch.zeros(2, 3, 2, 8)
        observations[:, -1, 0, 1] = 1.0
        labels = torch.zeros(2, dtype=torch.long)

        output = model(
            observations=observations,
            labels=labels,
            route_mode="hard",
            compute_penalties={},
            return_trace=True,
        )

        assert output.logits.shape == (2, 3)
        assert output.trace is not None
        assert output.trace["final_sink_state"].shape == (2, 4)
        assert output.trace["final_readout_input"].shape == (2, 4)


def test_multiview_readouts_construct_and_run() -> None:
    for mode in ("multiview_concat", "multiview_query_gated", "multiview_cross_attention"):
        model = PacketRoutingModel(
            {
                "num_nodes": 2,
                "obs_dim": 8,
                "hidden_dim": 4,
                "num_classes": 3,
                "max_internal_steps": 1,
                "max_total_steps": 8,
                "adapter_rank": 0,
                "readout_mode": mode,
                "readout_base_mode": "query_gated",
                "readout_views": ["final_sink_state", "packet_state_query", "baseline_readout_input"],
                "readout_iter_steps": 2,
                "readout_adapter_mode": "affine",
            }
        )
        observations = torch.zeros(2, 3, 2, 8)
        observations[:, -1, 0, 1] = 1.0
        labels = torch.zeros(2, dtype=torch.long)

        output = model(
            observations=observations,
            labels=labels,
            route_mode="hard",
            compute_penalties={},
            return_trace=True,
        )

        assert output.logits.shape == (2, 3)
        assert output.trace is not None
        assert output.trace["sink_state_query"].shape == (2, 4)
        assert output.trace["packet_state_query"].shape == (2, 4)
        assert output.trace["baseline_readout_input"].shape == (2, 4)
        assert output.trace["final_readout_input"].shape == (2, 4)


def test_delay_hold_mode_preserves_packet_state() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "delay_state_mode": "hold",
        }
    )
    current = torch.full((1, 2, 4), 2.0)
    updated = torch.full((1, 2, 4), -3.0)

    delayed = model._delay_packet_state(current_state=current, updated_state=updated, delay_retain=None)

    assert torch.equal(delayed, current)


def test_final_query_margin_and_focal_shaping_only_affect_final_query_examples() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.2, -1.0],
            [0.2, 0.1, -0.4],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    final_query_mask = torch.tensor([1, 0], dtype=torch.long)

    base_loss, base_shape = compute_task_classification_loss(
        logits,
        labels,
        final_query_mask=final_query_mask,
    )
    margin_loss, margin_shape = compute_task_classification_loss(
        logits,
        labels,
        final_query_mask=final_query_mask,
        final_query_shaping_mode="margin",
        final_query_shaping_weight=1.0,
        final_query_margin=4.0,
    )
    focal_loss, focal_shape = compute_task_classification_loss(
        logits,
        labels,
        final_query_mask=final_query_mask,
        final_query_shaping_mode="focal",
        final_query_shaping_weight=0.5,
        final_query_focal_gamma=2.0,
    )

    assert torch.isclose(base_shape, torch.zeros_like(base_shape))
    assert margin_shape.item() > 0.0
    assert focal_shape.item() > 0.0
    assert margin_loss.item() > base_loss.item()
    assert focal_loss.item() > base_loss.item()

    no_final_query_loss, no_final_query_shape = compute_task_classification_loss(
        logits,
        labels,
        final_query_mask=torch.zeros_like(final_query_mask),
        final_query_shaping_mode="margin",
        final_query_shaping_weight=1.0,
        final_query_margin=4.0,
    )
    assert torch.isclose(no_final_query_shape, torch.zeros_like(no_final_query_shape))
    assert torch.isclose(no_final_query_loss, base_loss)


def test_delay_adaptive_blend_interpolates_packet_state() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "delay_state_mode": "adaptive_blend",
        }
    )
    current = torch.full((1, 2, 4), 2.0)
    updated = torch.full((1, 2, 4), -2.0)
    delay_retain = torch.tensor([[[0.75], [0.25]]])

    delayed = model._delay_packet_state(
        current_state=current,
        updated_state=updated,
        delay_retain=delay_retain,
    )

    expected = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]],
    )
    assert torch.allclose(delayed, expected)


def test_packet_memory_write_and_read_round_trip() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "packet_memory_slots": 2,
            "packet_memory_dim": 3,
        }
    )
    with torch.no_grad():
        for module in [model.memory_read_mlp, model.memory_write_mlp]:
            for parameter in module.parameters():
                parameter.zero_()
        model.memory_write_slots.weight.zero_()
        model.memory_write_slots.bias.copy_(torch.tensor([10.0, -10.0]))
        model.memory_write_gate.weight.zero_()
        model.memory_write_gate.bias.fill_(10.0)
        model.memory_write_value.weight.zero_()
        model.memory_write_value.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))

        model.memory_read_slots.weight.zero_()
        model.memory_read_slots.bias.copy_(torch.tensor([10.0, -10.0]))
        model.memory_read_gate.weight.zero_()
        model.memory_read_gate.bias.fill_(10.0)

    packet_memory = torch.zeros(1, 2, 2, 3)
    packet_state = torch.zeros(1, 2, 4)
    node_state = torch.zeros(1, 2, 4)
    observations = torch.zeros(1, 2, 8)

    updated_memory, write_gate, write_weights = model._write_packet_memory(
        packet_memory=packet_memory,
        packet_state=packet_state,
        node_state=node_state,
        observations=observations,
    )
    read_state, read_gate, read_weights = model._read_packet_memory(
        packet_memory=updated_memory,
        packet_state=packet_state,
        node_state=node_state,
        observations=observations,
    )

    expected_value = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(write_gate.squeeze(-1), torch.ones(1, 2), atol=1e-4)
    assert torch.allclose(write_weights[..., 0], torch.ones(1, 2), atol=1e-4)
    assert torch.allclose(updated_memory[0, 0, 0], expected_value, atol=2e-4)
    assert torch.allclose(updated_memory[0, 0, 1], torch.zeros(3), atol=1e-4)
    assert torch.allclose(read_gate.squeeze(-1), torch.ones(1, 2), atol=1e-4)
    assert torch.allclose(read_weights[..., 0], torch.ones(1, 2), atol=1e-4)
    assert torch.allclose(read_state[0, 0], expected_value, atol=2e-4)


def test_control_state_sticky_retains_set_signal() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "control_state_dim": 1,
            "control_state_mode": "sticky",
        }
    )
    with torch.no_grad():
        for parameter in model.control_update_mlp.parameters():
            parameter.zero_()
        model.control_set_gate.weight.zero_()
        model.control_set_gate.bias.fill_(10.0)

    control_state = torch.zeros(1, 2, 1)
    packet_state = torch.zeros(1, 2, 4)
    node_state = torch.zeros(1, 2, 4)
    observations = torch.zeros(1, 2, 8)

    updated, _, _ = model._update_control_state(
        control_state=control_state,
        packet_state=packet_state,
        node_state=node_state,
        observations=observations,
    )
    assert torch.allclose(updated, torch.ones_like(updated), atol=1e-4)

    with torch.no_grad():
        model.control_set_gate.bias.fill_(-10.0)
    retained, _, _ = model._update_control_state(
        control_state=updated,
        packet_state=packet_state,
        node_state=node_state,
        observations=observations,
    )
    assert torch.allclose(retained, torch.ones_like(retained), atol=1e-4)


def test_control_state_set_clear_can_clear_signal() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "control_state_dim": 1,
            "control_state_mode": "set_clear",
        }
    )
    with torch.no_grad():
        for parameter in model.control_update_mlp.parameters():
            parameter.zero_()
        model.control_set_gate.weight.zero_()
        model.control_set_gate.bias.fill_(-10.0)
        model.control_clear_gate.weight.zero_()
        model.control_clear_gate.bias.fill_(10.0)

    control_state = torch.ones(1, 2, 1)
    packet_state = torch.zeros(1, 2, 4)
    node_state = torch.zeros(1, 2, 4)
    observations = torch.zeros(1, 2, 8)

    updated, _, clear = model._update_control_state(
        control_state=control_state,
        packet_state=packet_state,
        node_state=node_state,
        observations=observations,
    )
    assert clear is not None
    assert torch.allclose(updated, torch.zeros_like(updated), atol=1e-4)


def test_wait_loss_and_anti_exit_can_run_together_without_mask_collision() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 16,
            "adapter_rank": 0,
            "routing_head_mode": "wait_act",
            "control_state_dim": 4,
            "control_state_mode": "set_clear",
        }
    )
    observations = torch.zeros(2, 8, 2, 8)
    labels = torch.zeros(2, dtype=torch.long)
    wait_targets = torch.zeros(2, 8)
    wait_mask = torch.zeros(2, 8)
    anti_exit_mask = torch.zeros(2, 8)
    wait_targets[0, 2:7] = 1.0
    wait_mask[0, 2:8] = 1.0
    anti_exit_mask[0, 2:7] = 1.0

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard_st",
        compute_penalties={},
        wait_targets=wait_targets,
        wait_mask=wait_mask,
        wait_weight=0.5,
        anti_exit_mask=anti_exit_mask,
        anti_exit_weight=0.5,
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.stats["wait_loss"]).all()
    assert torch.isfinite(output.stats["anti_exit_loss"]).all()


def test_direct_release_gate_controls_wait_vs_act_decision() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 2,
            "max_internal_steps": 1,
            "max_total_steps": 16,
            "adapter_rank": 0,
            "routing_head_mode": "wait_act",
            "release_gate_mode": "direct",
            "release_gate_scale": 2.0,
        }
    )
    act_logits = torch.tensor([[[4.0, -4.0], [4.0, -4.0]]])
    wait_logit = torch.tensor([[[6.0], [-6.0]]])
    release_logit = torch.tensor([[[-6.0], [6.0]]])

    composed = model._compose_routing_logits(
        flat_logits=None,
        act_logits=act_logits,
        wait_logit=wait_logit,
        release_logit=release_logit,
        control_state=torch.zeros(1, 2, 0),
    )

    actions = composed.argmax(dim=-1)
    assert actions.tolist() == [[2, 0]]


def test_cosine_readout_head_constructs_and_runs() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 3,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "readout_mode": "multiview_query_gated",
            "readout_base_mode": "query_gated",
            "readout_views": ["final_sink_state", "packet_state_query", "baseline_readout_input"],
            "readout_head_mode": "cosine",
            "readout_metric_dim": 6,
            "readout_prototype_pull_weight": 0.1,
        }
    )
    observations = torch.zeros(2, 3, 2, 8)
    observations[:, -1, 0, 1] = 1.0
    labels = torch.tensor([0, 1], dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
        return_trace=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.trace is not None
    assert output.trace["final_readout_input"].shape == (2, 4)
    assert "prototype_pull_loss" in output.stats
    assert output.stats["prototype_pull_loss"].shape == (2,)


def test_cosine_readout_prototype_pull_loss_respects_sample_weights() -> None:
    head = CosineReadoutHead(input_dim=4, num_classes=3, metric_dim=4, init_scale=8.0)
    with torch.no_grad():
        head.readout_prototypes.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2], dtype=torch.long)
    weights = torch.tensor([1.0, 0.0])

    full_loss = head.prototype_pull_loss(features, labels)
    loss = head.prototype_pull_loss(features, labels, sample_weights=weights)

    single_loss = head.prototype_pull_loss(features[:1], labels[:1])

    assert torch.isclose(loss, single_loss, atol=1e-6)
    assert not torch.isclose(full_loss, single_loss, atol=1e-6)


def test_mixture_readout_head_constructs_and_runs() -> None:
    model = PacketRoutingModel(
        {
            "num_nodes": 2,
            "obs_dim": 8,
            "hidden_dim": 4,
            "num_classes": 3,
            "max_internal_steps": 1,
            "max_total_steps": 8,
            "adapter_rank": 0,
            "readout_mode": "multiview_query_gated",
            "readout_base_mode": "query_gated",
            "readout_views": ["final_sink_state", "packet_state_query", "baseline_readout_input"],
            "readout_head_mode": "mixture",
            "readout_mixture_num_heads": 3,
            "readout_mixture_gate_source": "input_query",
            "readout_mixture_gate_hidden_dim": 5,
            "readout_mixture_branch_hidden_dim": 6,
            "readout_mixture_balance_weight": 0.1,
        }
    )
    observations = torch.zeros(2, 3, 2, 8)
    observations[:, -1, 0, 1] = 1.0
    labels = torch.tensor([0, 1], dtype=torch.long)

    output = model(
        observations=observations,
        labels=labels,
        route_mode="hard",
        compute_penalties={},
        return_trace=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.trace is not None
    assert output.trace["final_readout_input"].shape == (2, 4)
    assert "mixture_balance_loss" in output.stats
    assert "mixture_gate_entropy" in output.stats
    assert "mixture_top1_weight" in output.stats
    assert output.stats["mixture_balance_loss"].shape == (2,)


def test_mixture_readout_head_weights_form_distribution() -> None:
    head = MixtureReadoutHead(
        input_dim=4,
        num_classes=3,
        query_dim=6,
        num_heads=3,
        branch_hidden_dim=5,
        gate_source="input_query",
        gate_hidden_dim=7,
    )
    features = torch.randn(2, 4)
    query = torch.randn(2, 6)

    logits = head(features, query_obs=query)

    assert logits.shape == (2, 3)
    assert head.last_mixture_weights is not None
    assert torch.allclose(head.last_mixture_weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert head.balance_loss().item() >= 0.0
    assert head.entropy().item() >= 0.0
