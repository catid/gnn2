from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_FORWARD = 0
ACTION_EXIT = 1
ACTION_DELAY = 2


@dataclass
class RoutingForwardOutput:
    logits: torch.Tensor
    sink_state: torch.Tensor
    loss: torch.Tensor
    route_loss: torch.Tensor
    task_loss: torch.Tensor
    stats: dict[str, torch.Tensor]


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class NodeCore(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_nodes: int,
        adapter_rank: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.meta_proj = nn.Linear(4, hidden_dim)
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        self.pre = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.state_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.packet_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.packet_adapter = (
            LowRankAdapter(hidden_dim, adapter_rank) if adapter_rank > 0 else None
        )
        self.router_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.router_out = nn.Linear(hidden_dim, 3)

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
        obs_features = self.obs_proj(observations)
        node_features = self.node_embed(node_index)
        meta = torch.stack(
            [age_fraction, time_fraction, remaining_fraction, node_index.float()],
            dim=-1,
        )
        meta[..., -1] = meta[..., -1] / max(1.0, float(self.node_embed.num_embeddings - 1))
        meta_features = self.meta_proj(meta)

        merged = torch.cat(
            [packet_state, node_state, obs_features, node_features, meta_features],
            dim=-1,
        )
        proposal = self.pre(merged)

        batch_shape = packet_state.shape[:-1]
        node_state_next = self.state_cell(
            proposal.reshape(-1, self.hidden_dim),
            node_state.reshape(-1, self.hidden_dim),
        ).reshape(*batch_shape, self.hidden_dim)

        packet_features = torch.cat(
            [packet_state, node_state_next, obs_features, node_features], dim=-1
        )
        packet_delta = self.packet_mlp(packet_features)
        if self.packet_adapter is not None:
            packet_delta = packet_delta + self.packet_adapter(packet_delta)
        packet_next = packet_state + packet_delta

        router_hidden = self.router_mlp(packet_features)
        logits = self.router_out(router_hidden)
        return node_state_next, packet_next, logits


class PacketRoutingModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_nodes = int(config["num_nodes"])
        self.obs_dim = int(config["obs_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.num_classes = int(config["num_classes"])
        self.max_internal_steps = int(config.get("max_internal_steps", self.num_nodes))
        self.max_total_steps = int(config.get("max_total_steps", 4096))
        self.adapter_rank = int(config.get("adapter_rank", 0))
        self.core = NodeCore(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            adapter_rank=self.adapter_rank,
        )
        self.sink_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def es_parameter_names(self, include_adapters: bool) -> list[str]:
        allowed = [
            "core.router_mlp",
            "core.router_out",
        ]
        if include_adapters:
            allowed.append("core.packet_adapter")
        names: list[str] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in allowed):
                names.append(name)
        return names

    def forward(
        self,
        observations: torch.Tensor,
        labels: torch.Tensor,
        route_mode: str,
        compute_penalties: dict[str, float] | None = None,
        temperature: float = 1.0,
        estimator: str = "straight_through",
        truncate_bptt_steps: int = 0,
    ) -> RoutingForwardOutput:
        batch_size, seq_len, num_nodes, _ = observations.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, received {num_nodes}."
            )

        device = observations.device
        dtype = observations.dtype
        eps = torch.tensor(1e-8, device=device, dtype=dtype)
        node_indices = torch.arange(self.num_nodes, device=device).view(1, self.num_nodes)

        node_states = torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device, dtype=dtype)
        packet_states = torch.zeros_like(node_states)
        packet_masses = torch.zeros(batch_size, self.num_nodes, device=device, dtype=dtype)
        packet_masses[:, 0] = 1.0
        packet_ages = torch.zeros(batch_size, self.num_nodes, 1, device=device, dtype=dtype)
        sink_state = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

        hops = torch.zeros(batch_size, device=device, dtype=dtype)
        delays = torch.zeros(batch_size, device=device, dtype=dtype)
        exits = torch.zeros(batch_size, device=device, dtype=dtype)
        ttl_fail = torch.zeros(batch_size, device=device, dtype=dtype)
        first_exit_time = torch.zeros(batch_size, device=device, dtype=dtype)
        exit_mass_so_far = torch.zeros(batch_size, device=device, dtype=dtype)
        early_exit_mass = torch.zeros(batch_size, device=device, dtype=dtype)

        for time_index in range(seq_len):
            obs_t = observations[:, time_index]
            carry_mass = torch.zeros_like(packet_masses)
            carry_state_sum = torch.zeros_like(packet_states)
            carry_age_sum = torch.zeros_like(packet_ages)

            for _ in range(self.max_internal_steps):
                mass = packet_masses
                if float(mass.max().detach().cpu()) <= 0.0:
                    break

                age_fraction = (packet_ages.squeeze(-1) / max(1, self.max_total_steps)).clamp_(0.0, 1.0)
                time_fraction = torch.full_like(
                    mass,
                    float(time_index) / max(1, seq_len - 1),
                )
                remaining_fraction = (1.0 - age_fraction).clamp_(0.0, 1.0)

                node_state_next, packet_next, logits = self.core(
                    packet_state=packet_states,
                    node_state=node_states,
                    observations=obs_t,
                    node_index=node_indices.expand(batch_size, -1),
                    age_fraction=age_fraction,
                    time_fraction=time_fraction,
                    remaining_fraction=remaining_fraction,
                )

                actions = self._route_actions(
                    logits=logits,
                    route_mode=route_mode,
                    temperature=temperature,
                    estimator=estimator,
                )

                node_states = node_states + mass.unsqueeze(-1) * (node_state_next - node_states)

                weighted_packet = packet_next * mass.unsqueeze(-1)
                forward_mass = mass * actions[..., ACTION_FORWARD]
                exit_mass = mass * actions[..., ACTION_EXIT]
                delay_mass = mass * actions[..., ACTION_DELAY]
                next_age = packet_ages + 1.0

                hops = hops + forward_mass.sum(dim=1)
                delays = delays + delay_mass.sum(dim=1)

                exit_features = self.sink_proj(packet_next)
                exit_contrib = (exit_mass.unsqueeze(-1) * exit_features).sum(dim=1)
                sink_state = sink_state + exit_contrib
                exit_added = exit_mass.sum(dim=1)
                newly_exited = (exit_mass_so_far <= 0.0) & (exit_added > 0.0)
                first_exit_time = torch.where(
                    newly_exited,
                    torch.full_like(first_exit_time, float(time_index)),
                    first_exit_time,
                )
                if time_index == 0:
                    early_exit_mass = early_exit_mass + exit_added
                exit_mass_so_far = exit_mass_so_far + exit_added
                exits = exits + exit_added

                carry_mass = carry_mass + delay_mass
                carry_state_sum = carry_state_sum + delay_mass.unsqueeze(-1) * packet_next
                carry_age_sum = carry_age_sum + delay_mass.unsqueeze(-1) * next_age

                next_mass = torch.zeros_like(packet_masses)
                next_state_sum = torch.zeros_like(packet_states)
                next_age_sum = torch.zeros_like(packet_ages)
                if self.num_nodes > 1:
                    next_mass[:, 1:] = next_mass[:, 1:] + forward_mass[:, :-1]
                    next_state_sum[:, 1:, :] = next_state_sum[:, 1:, :] + weighted_packet[:, :-1, :] * actions[:, :-1, ACTION_FORWARD].unsqueeze(-1)
                    next_age_sum[:, 1:, :] = next_age_sum[:, 1:, :] + next_age[:, :-1, :] * forward_mass[:, :-1].unsqueeze(-1)

                terminal_forward_mass = forward_mass[:, -1]
                sink_state = sink_state + terminal_forward_mass.unsqueeze(-1) * exit_features[:, -1, :]
                newly_terminal = (exit_mass_so_far <= 0.0) & (terminal_forward_mass > 0.0)
                first_exit_time = torch.where(
                    newly_terminal,
                    torch.full_like(first_exit_time, float(time_index)),
                    first_exit_time,
                )
                if time_index == 0:
                    early_exit_mass = early_exit_mass + terminal_forward_mass
                exit_mass_so_far = exit_mass_so_far + terminal_forward_mass
                exits = exits + terminal_forward_mass

                packet_masses = next_mass
                packet_states = self._normalize_state(next_state_sum, next_mass, eps)
                packet_ages = self._normalize_state(next_age_sum, next_mass, eps)

            forced_delay = packet_masses
            if float(forced_delay.max().detach().cpu()) > 0.0:
                delays = delays + forced_delay.sum(dim=1)
                carry_mass = carry_mass + forced_delay
                carry_state_sum = carry_state_sum + forced_delay.unsqueeze(-1) * packet_states
                carry_age_sum = carry_age_sum + forced_delay.unsqueeze(-1) * (packet_ages + 1.0)

            packet_masses = carry_mass
            packet_states = self._normalize_state(carry_state_sum, carry_mass, eps)
            packet_ages = self._normalize_state(carry_age_sum, carry_mass, eps)

            if truncate_bptt_steps > 0 and (time_index + 1) % truncate_bptt_steps == 0:
                node_states = node_states.detach()
                packet_states = packet_states.detach()
                packet_masses = packet_masses.detach()
                packet_ages = packet_ages.detach()
                sink_state = sink_state.detach()

        residual_mass = packet_masses.sum(dim=1)
        if float(residual_mass.max().detach().cpu()) > 0.0:
            sink_state = sink_state + (packet_masses.unsqueeze(-1) * self.sink_proj(packet_states)).sum(dim=1)
            ttl_fail = ttl_fail + residual_mass
            exits = exits + residual_mass
            newly_exited = (exit_mass_so_far <= 0.0) & (residual_mass > 0.0)
            first_exit_time = torch.where(
                newly_exited,
                torch.full_like(first_exit_time, float(seq_len - 1)),
                first_exit_time,
            )
            exit_mass_so_far = exit_mass_so_far + residual_mass

        logits = self.readout(sink_state)
        task_loss = F.cross_entropy(logits, labels)
        penalties = compute_penalties or {}
        route_loss = (
            float(penalties.get("hops", 0.0)) * hops.mean()
            + float(penalties.get("delays", 0.0)) * delays.mean()
            + float(penalties.get("ttl_fail", 0.0)) * ttl_fail.mean()
        )
        total_loss = task_loss + route_loss

        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float()
        exit_time = torch.where(
            exit_mass_so_far > 0.0,
            first_exit_time,
            torch.full_like(first_exit_time, float(seq_len - 1)),
        )

        stats = {
            "accuracy": accuracy,
            "hops": hops,
            "delays": delays,
            "exits": exits,
            "ttl_fail": ttl_fail,
            "exit_time": exit_time,
            "early_exit_mass": early_exit_mass,
            "compute": hops + delays + exits,
        }
        return RoutingForwardOutput(
            logits=logits,
            sink_state=sink_state,
            loss=total_loss,
            route_loss=route_loss,
            task_loss=task_loss,
            stats=stats,
        )

    @staticmethod
    def _normalize_state(
        weighted_state: torch.Tensor,
        mass: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        denom = mass.unsqueeze(-1).clamp_min(float(eps))
        normalized = weighted_state / denom
        return torch.where(mass.unsqueeze(-1) > 0.0, normalized, torch.zeros_like(weighted_state))

    @staticmethod
    def _route_actions(
        logits: torch.Tensor,
        route_mode: str,
        temperature: float,
        estimator: str,
    ) -> torch.Tensor:
        if route_mode == "soft":
            return F.softmax(logits / temperature, dim=-1)

        if route_mode == "hard":
            indices = logits.argmax(dim=-1)
            return F.one_hot(indices, num_classes=3).float()

        if route_mode == "hard_st":
            if estimator == "gumbel":
                return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
            soft = F.softmax(logits / temperature, dim=-1)
            hard = F.one_hot(soft.argmax(dim=-1), num_classes=3).float()
            return hard + soft - soft.detach()

        raise ValueError(f"Unknown route mode: {route_mode}")
