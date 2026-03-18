from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_FORWARD = 0
ACTION_EXIT = 1
ACTION_DELAY = 2

ACTION_NAMES = {
    ACTION_FORWARD: "forward",
    ACTION_EXIT: "exit",
    ACTION_DELAY: "delay",
}


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
        packet_update: str = "residual",
        delay_state_mode: str = "updated",
        delay_gate_bias: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.packet_update = packet_update
        self.delay_state_mode = delay_state_mode
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
        self.packet_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.packet_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.packet_gate[1].bias, 1.5)
        self.packet_adapter = (
            LowRankAdapter(hidden_dim, adapter_rank) if adapter_rank > 0 else None
        )
        self.router_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.router_out = nn.Linear(hidden_dim, 3)
        self.delay_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.delay_gate[3].bias, delay_gate_bias)

    def forward(
        self,
        packet_state: torch.Tensor,
        node_state: torch.Tensor,
        observations: torch.Tensor,
        node_index: torch.Tensor,
        age_fraction: torch.Tensor,
        time_fraction: torch.Tensor,
        remaining_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        packet_proposal = self.packet_mlp(packet_features)
        if self.packet_adapter is not None:
            packet_proposal = packet_proposal + self.packet_adapter(packet_proposal)
        if self.packet_update == "gru":
            packet_next = self.packet_cell(
                packet_proposal.reshape(-1, self.hidden_dim),
                packet_state.reshape(-1, self.hidden_dim),
            ).reshape(*batch_shape, self.hidden_dim)
        elif self.packet_update == "gated_gru":
            packet_candidate = self.packet_cell(
                packet_proposal.reshape(-1, self.hidden_dim),
                packet_state.reshape(-1, self.hidden_dim),
            ).reshape(*batch_shape, self.hidden_dim)
            retain = self.packet_gate(packet_features)
            packet_next = retain * packet_state + (1.0 - retain) * packet_candidate
        else:
            packet_next = packet_state + packet_proposal

        router_hidden = self.router_mlp(packet_features)
        logits = self.router_out(router_hidden)
        delay_retain = None
        if self.delay_state_mode == "adaptive_blend":
            delay_retain = self.delay_gate(packet_features)
        return node_state_next, packet_next, logits, delay_retain


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
        self.packet_update = str(config.get("packet_update", "residual"))
        self.delay_state_mode = str(config.get("delay_state_mode", "updated"))
        self.delay_gate_bias = float(config.get("delay_gate_bias", 2.0))
        self.core = NodeCore(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            adapter_rank=self.adapter_rank,
            packet_update=self.packet_update,
            delay_state_mode=self.delay_state_mode,
            delay_gate_bias=self.delay_gate_bias,
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
        forced_actions: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
        oracle_actions: torch.Tensor | None = None,
        oracle_action_mask: torch.Tensor | None = None,
        oracle_route_weight: float = 0.0,
        delay_write_targets: torch.Tensor | None = None,
        delay_write_mask: torch.Tensor | None = None,
        delay_write_weight: float = 0.0,
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

        action_totals = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        decision_mass_total = torch.zeros(batch_size, device=device, dtype=dtype)
        route_entropy_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        route_conf_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        route_weight_total = torch.zeros(batch_size, device=device, dtype=dtype)
        transition_totals = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        prev_action_summary = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        prev_has_action = torch.zeros(batch_size, device=device, dtype=torch.bool)

        mailbox_mass_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        mailbox_peak = torch.zeros(batch_size, device=device, dtype=dtype)
        mailbox_steps = torch.zeros(batch_size, device=device, dtype=dtype)
        age_mass_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        age_weight_total = torch.zeros(batch_size, device=device, dtype=dtype)
        max_age = torch.zeros(batch_size, device=device, dtype=dtype)
        exit_age_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        exit_age_mass = torch.zeros(batch_size, device=device, dtype=dtype)
        delay_retain_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        delay_retain_weight = torch.zeros(batch_size, device=device, dtype=dtype)

        oracle_route_loss = torch.zeros((), device=device, dtype=dtype)
        oracle_weight_total = torch.zeros((), device=device, dtype=dtype)
        delay_write_loss = torch.zeros((), device=device, dtype=dtype)
        delay_write_total = torch.zeros((), device=device, dtype=dtype)

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

                core_output = self.core(
                    packet_state=packet_states,
                    node_state=node_states,
                    observations=obs_t,
                    node_index=node_indices.expand(batch_size, -1),
                    age_fraction=age_fraction,
                    time_fraction=time_fraction,
                    remaining_fraction=remaining_fraction,
                )
                if len(core_output) == 3:
                    node_state_next, packet_next, logits = core_output
                    delay_retain = None
                else:
                    node_state_next, packet_next, logits, delay_retain = core_output

                masked_logits = logits
                if action_masks is not None:
                    masked_logits = self._apply_action_mask(masked_logits, action_masks[:, time_index])

                router_probs = F.softmax(masked_logits / max(temperature, 1e-6), dim=-1)
                router_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1)
                router_conf = router_probs.max(dim=-1).values
                active_mass = mass.sum(dim=1)
                route_entropy_sum = route_entropy_sum + (router_entropy * mass).sum(dim=1)
                route_conf_sum = route_conf_sum + (router_conf * mass).sum(dim=1)
                route_weight_total = route_weight_total + active_mass

                age_values = packet_ages.squeeze(-1)
                age_mass_sum = age_mass_sum + (age_values * mass).sum(dim=1)
                age_weight_total = age_weight_total + active_mass
                max_age = torch.maximum(max_age, age_values.max(dim=1).values)

                if oracle_actions is not None and oracle_action_mask is not None and oracle_route_weight > 0.0:
                    target = oracle_actions[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    step_mask = oracle_action_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    per_node_ce = F.cross_entropy(
                        masked_logits.reshape(-1, 3),
                        target.reshape(-1),
                        reduction="none",
                    ).reshape(batch_size, self.num_nodes)
                    weight = step_mask * mass
                    oracle_route_loss = oracle_route_loss + (per_node_ce * weight).sum()
                    oracle_weight_total = oracle_weight_total + weight.sum()
                if (
                    delay_retain is not None
                    and delay_write_targets is not None
                    and delay_write_mask is not None
                    and delay_write_weight > 0.0
                ):
                    write_target = delay_write_targets[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    write_mask = delay_write_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    write_prob = (1.0 - delay_retain.squeeze(-1).float()).clamp(1e-4, 1.0 - 1e-4)
                    write_logits = torch.logit(write_prob)
                    per_node_bce = F.binary_cross_entropy_with_logits(
                        write_logits,
                        write_target.float(),
                        reduction="none",
                    ).to(dtype)
                    weight = (write_mask * mass).float()
                    delay_write_loss = delay_write_loss + (per_node_bce * weight).sum()
                    delay_write_total = delay_write_total + weight.sum()

                actions = self._route_actions(
                    logits=masked_logits,
                    route_mode=route_mode,
                    temperature=temperature,
                    estimator=estimator,
                )
                if forced_actions is not None:
                    forced_step = forced_actions[:, time_index]
                    valid = forced_step >= 0
                    if valid.any():
                        forced = F.one_hot(forced_step.clamp_min(0), num_classes=3).to(dtype)
                        actions = torch.where(
                            valid.view(batch_size, 1, 1),
                            forced.view(batch_size, 1, 3).expand(-1, self.num_nodes, -1),
                            actions,
                        )

                action_summary = (mass.unsqueeze(-1) * actions).sum(dim=1)
                action_totals = action_totals + action_summary
                decision_mass_total = decision_mass_total + action_summary.sum(dim=-1)
                action_norm = action_summary / action_summary.sum(dim=-1, keepdim=True).clamp_min(float(eps))
                update_transition = prev_has_action & (action_summary.sum(dim=-1) > 0.0)
                if update_transition.any():
                    transition_totals[update_transition] = transition_totals[update_transition] + torch.einsum(
                        "bi,bj->bij",
                        prev_action_summary[update_transition],
                        action_norm[update_transition],
                    )
                prev_action_summary = torch.where(
                    (action_summary.sum(dim=-1) > 0.0).view(batch_size, 1),
                    action_norm,
                    prev_action_summary,
                )
                prev_has_action = prev_has_action | (action_summary.sum(dim=-1) > 0.0)

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
                exit_age_sum = exit_age_sum + (exit_mass * age_values).sum(dim=1)
                exit_age_mass = exit_age_mass + exit_added
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
                delayed_packet = self._delay_packet_state(
                    current_state=packet_states,
                    updated_state=packet_next,
                    delay_retain=delay_retain,
                )
                carry_state_sum = carry_state_sum + delay_mass.unsqueeze(-1) * delayed_packet
                carry_age_sum = carry_age_sum + delay_mass.unsqueeze(-1) * next_age
                if delay_retain is not None:
                    delay_retain_sum = delay_retain_sum + (delay_retain.squeeze(-1) * delay_mass).sum(dim=1)
                    delay_retain_weight = delay_retain_weight + delay_mass.sum(dim=1)

                next_mass = torch.zeros_like(packet_masses)
                next_state_sum = torch.zeros_like(packet_states)
                next_age_sum = torch.zeros_like(packet_ages)
                if self.num_nodes > 1:
                    next_mass[:, 1:] = next_mass[:, 1:] + forward_mass[:, :-1]
                    next_state_sum[:, 1:, :] = next_state_sum[:, 1:, :] + weighted_packet[:, :-1, :] * actions[:, :-1, ACTION_FORWARD].unsqueeze(-1)
                    next_age_sum[:, 1:, :] = next_age_sum[:, 1:, :] + next_age[:, :-1, :] * forward_mass[:, :-1].unsqueeze(-1)

                terminal_forward_mass = forward_mass[:, -1]
                sink_state = sink_state + terminal_forward_mass.unsqueeze(-1) * exit_features[:, -1, :]
                exit_age_sum = exit_age_sum + terminal_forward_mass * age_values[:, -1]
                exit_age_mass = exit_age_mass + terminal_forward_mass
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
            mailbox_load = carry_mass.sum(dim=1)
            mailbox_mass_sum = mailbox_mass_sum + mailbox_load
            mailbox_peak = torch.maximum(mailbox_peak, mailbox_load)
            mailbox_steps = mailbox_steps + (mailbox_load >= 0.0).float()

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
            exit_age_sum = exit_age_sum + (packet_ages.squeeze(-1) * packet_masses).sum(dim=1)
            exit_age_mass = exit_age_mass + residual_mass
            newly_exited = (exit_mass_so_far <= 0.0) & (residual_mass > 0.0)
            first_exit_time = torch.where(
                newly_exited,
                torch.full_like(first_exit_time, float(seq_len - 1)),
                first_exit_time,
            )
            exit_mass_so_far = exit_mass_so_far + residual_mass

        logits = self.readout(sink_state)
        task_loss = F.cross_entropy(logits, labels)
        penalty_hops = float(compute_penalties.get("hops", 0.0)) * hops.mean() if compute_penalties else torch.zeros((), device=device, dtype=dtype)
        penalty_delays = float(compute_penalties.get("delays", 0.0)) * delays.mean() if compute_penalties else torch.zeros((), device=device, dtype=dtype)
        penalty_ttl = float(compute_penalties.get("ttl_fail", 0.0)) * ttl_fail.mean() if compute_penalties else torch.zeros((), device=device, dtype=dtype)
        route_penalty = penalty_hops + penalty_delays + penalty_ttl
        oracle_route_term = torch.zeros((), device=device, dtype=dtype)
        if oracle_route_weight > 0.0 and float(oracle_weight_total.detach().cpu()) > 0.0:
            oracle_route_term = oracle_route_weight * (oracle_route_loss / oracle_weight_total.clamp_min(float(eps)))
        delay_write_term = torch.zeros((), device=device, dtype=dtype)
        if delay_write_weight > 0.0 and float(delay_write_total.detach().cpu()) > 0.0:
            delay_write_term = delay_write_weight * (delay_write_loss / delay_write_total.clamp_min(float(eps)))
        total_loss = task_loss + route_penalty + oracle_route_term + delay_write_term

        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float()
        exit_time = torch.where(
            exit_mass_so_far > 0.0,
            first_exit_time,
            torch.full_like(first_exit_time, float(seq_len - 1)),
        )

        decision_denom = decision_mass_total.clamp_min(float(eps)).unsqueeze(-1)
        action_rates = action_totals / decision_denom
        transition_total = transition_totals.sum(dim=(1, 2), keepdim=True).clamp_min(float(eps))
        transition_norm = transition_totals / transition_total

        stats = {
            "accuracy": accuracy,
            "hops": hops,
            "delays": delays,
            "exits": exits,
            "ttl_fail": ttl_fail,
            "exit_time": exit_time,
            "early_exit_mass": early_exit_mass,
            "compute": hops + delays + exits,
            "forward_rate": action_rates[:, ACTION_FORWARD],
            "exit_rate": action_rates[:, ACTION_EXIT],
            "delay_rate": action_rates[:, ACTION_DELAY],
            "route_entropy": route_entropy_sum / route_weight_total.clamp_min(float(eps)),
            "router_confidence": route_conf_sum / route_weight_total.clamp_min(float(eps)),
            "mailbox_occupancy": mailbox_mass_sum / mailbox_steps.clamp_min(1.0),
            "mailbox_peak": mailbox_peak,
            "packet_age_mean": age_mass_sum / age_weight_total.clamp_min(float(eps)),
            "packet_age_max": max_age,
            "exit_age_mean": exit_age_sum / exit_age_mass.clamp_min(float(eps)),
            "decision_count": decision_mass_total,
            "delay_retain_mean": delay_retain_sum / delay_retain_weight.clamp_min(float(eps)),
            "delay_write_mean": 1.0 - (delay_retain_sum / delay_retain_weight.clamp_min(float(eps))),
            "penalty_hops": torch.full_like(accuracy, float(penalty_hops.detach().item())),
            "penalty_delays": torch.full_like(accuracy, float(penalty_delays.detach().item())),
            "penalty_ttl": torch.full_like(accuracy, float(penalty_ttl.detach().item())),
            "oracle_route_loss": torch.full_like(accuracy, float(oracle_route_term.detach().item())),
            "delay_write_loss": torch.full_like(accuracy, float(delay_write_term.detach().item())),
        }
        for src_index, src_name in ACTION_NAMES.items():
            for dst_index, dst_name in ACTION_NAMES.items():
                stats[f"transition_{src_name}_to_{dst_name}"] = transition_norm[:, src_index, dst_index]

        return RoutingForwardOutput(
            logits=logits,
            sink_state=sink_state,
            loss=total_loss,
            route_loss=route_penalty + oracle_route_term + delay_write_term,
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
    def _apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        expanded = action_mask.unsqueeze(1).expand(-1, logits.shape[1], -1)
        return logits.masked_fill(expanded <= 0.0, -1e9)

    def _delay_packet_state(
        self,
        current_state: torch.Tensor,
        updated_state: torch.Tensor,
        delay_retain: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.delay_state_mode == "hold":
            return current_state
        if self.delay_state_mode == "adaptive_blend" and delay_retain is not None:
            return delay_retain * current_state + (1.0 - delay_retain) * updated_state
        return updated_state

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
