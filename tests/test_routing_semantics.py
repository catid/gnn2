from __future__ import annotations

import torch
import torch.nn as nn

from src.models.packet_routing import PacketRoutingModel


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
