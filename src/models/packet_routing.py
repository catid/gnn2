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
    trace: dict[str, torch.Tensor] | None = None


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class AffineAdapter(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.scale) + self.bias


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
        routing_head_mode: str = "flat",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.packet_update = packet_update
        self.delay_state_mode = delay_state_mode
        self.routing_head_mode = routing_head_mode
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
        if self.routing_head_mode == "flat":
            self.router_out = nn.Linear(hidden_dim, 3)
            self.router_act_out = None
            self.router_wait_out = None
        else:
            self.router_out = None
            self.router_act_out = nn.Linear(hidden_dim, 2)
            self.router_wait_out = nn.Linear(hidden_dim, 1)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
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
        if self.routing_head_mode == "flat":
            logits = self.router_out(router_hidden)
            act_logits = None
            wait_logit = None
        else:
            logits = None
            act_logits = self.router_act_out(router_hidden)
            wait_logit = self.router_wait_out(router_hidden)
        delay_retain = None
        if self.delay_state_mode == "adaptive_blend":
            delay_retain = self.delay_gate(packet_features)
        return node_state_next, packet_next, logits, delay_retain, {
            "router_hidden": router_hidden,
            "act_logits": act_logits,
            "wait_logit": wait_logit,
        }


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
        self.routing_head_mode = str(config.get("routing_head_mode", "flat"))
        self.packet_memory_slots = int(config.get("packet_memory_slots", 0))
        self.packet_memory_dim = int(config.get("packet_memory_dim", self.hidden_dim))
        self.packet_memory_read_bias = float(config.get("packet_memory_read_bias", 1.0))
        self.packet_memory_write_bias = float(config.get("packet_memory_write_bias", -1.0))
        self.control_state_dim = int(config.get("control_state_dim", 0))
        self.control_state_mode = str(config.get("control_state_mode", "sticky"))
        self.control_input_scale = float(config.get("control_input_scale", 1.0))
        self.control_router_scale = float(config.get("control_router_scale", 1.0))
        self.control_wait_scale = float(config.get("control_wait_scale", 0.0))
        self.control_write_bias = float(config.get("control_write_bias", -3.0))
        self.control_clear_bias = float(config.get("control_clear_bias", -3.0))
        self.wait_state_dim = int(config.get("wait_state_dim", 0))
        self.wait_state_input_scale = float(config.get("wait_state_input_scale", 0.0))
        self.release_scale = float(config.get("release_scale", 0.0))
        self.release_gate_mode = str(config.get("release_gate_mode", "bias"))
        self.release_gate_scale = float(config.get("release_gate_scale", 1.0))
        self.readout_mode = str(config.get("readout_mode", "plain"))
        self.readout_base_mode = str(config.get("readout_base_mode", "plain"))
        self.readout_views = tuple(
            str(name)
            for name in config.get(
                "readout_views",
                ["final_sink_state", "packet_state_query"],
            )
        )
        self.readout_iter_steps = int(config.get("readout_iter_steps", 1))
        self.readout_view_dropout = float(config.get("readout_view_dropout", 0.0))
        self.readout_attention_heads = int(config.get("readout_attention_heads", 1))
        self.readout_adapter_mode = str(config.get("readout_adapter_mode", "none"))
        self.readout_adapter_rank = int(config.get("readout_adapter_rank", 0))
        self.readout_adapter_hidden_dim = int(
            config.get("readout_adapter_hidden_dim", self.hidden_dim)
        )
        self.multiview_readout_modes = {
            "multiview_concat",
            "multiview_query_gated",
            "multiview_query_film",
            "multiview_cross_attention",
        }
        self.core = NodeCore(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            adapter_rank=self.adapter_rank,
            packet_update=self.packet_update,
            delay_state_mode=self.delay_state_mode,
            delay_gate_bias=self.delay_gate_bias,
            routing_head_mode=self.routing_head_mode,
        )
        if self.packet_memory_slots > 0:
            memory_feature_dim = self.hidden_dim * 2 + self.obs_dim
            self.memory_read_mlp = nn.Sequential(
                nn.LayerNorm(memory_feature_dim),
                nn.Linear(memory_feature_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.memory_read_slots = nn.Linear(self.hidden_dim, self.packet_memory_slots)
            self.memory_read_gate = nn.Linear(self.hidden_dim, 1)
            nn.init.constant_(self.memory_read_gate.bias, self.packet_memory_read_bias)

            self.memory_write_mlp = nn.Sequential(
                nn.LayerNorm(memory_feature_dim),
                nn.Linear(memory_feature_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.memory_write_slots = nn.Linear(self.hidden_dim, self.packet_memory_slots)
            self.memory_write_gate = nn.Linear(self.hidden_dim, 1)
            nn.init.constant_(self.memory_write_gate.bias, self.packet_memory_write_bias)
            self.memory_write_value = nn.Linear(self.hidden_dim, self.packet_memory_dim)
            self.memory_input_proj = nn.Linear(self.packet_memory_dim, self.hidden_dim)
            self.memory_payload_head = nn.Sequential(
                nn.LayerNorm(self.packet_memory_dim),
                nn.Linear(self.packet_memory_dim, self.num_classes),
            )
        else:
            self.memory_read_mlp = None
            self.memory_read_slots = None
            self.memory_read_gate = None
            self.memory_write_mlp = None
            self.memory_write_slots = None
            self.memory_write_gate = None
            self.memory_write_value = None
            self.memory_input_proj = None
            self.memory_payload_head = None
        if self.control_state_dim > 0:
            control_feature_dim = self.hidden_dim * 2 + self.obs_dim
            self.control_update_mlp = nn.Sequential(
                nn.LayerNorm(control_feature_dim),
                nn.Linear(control_feature_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.control_set_gate = nn.Linear(self.hidden_dim, self.control_state_dim)
            nn.init.constant_(self.control_set_gate.bias, self.control_write_bias)
            if self.control_state_mode == "set_clear":
                self.control_clear_gate = nn.Linear(self.hidden_dim, self.control_state_dim)
                nn.init.constant_(self.control_clear_gate.bias, self.control_clear_bias)
            else:
                self.control_clear_gate = None
            self.control_input_proj = nn.Linear(self.control_state_dim, self.hidden_dim)
            self.control_router_out = nn.Linear(self.control_state_dim, 3, bias=False)
            self.control_wait_out = nn.Linear(self.control_state_dim, 1, bias=False)
            self.control_head = nn.Sequential(
                nn.LayerNorm(self.control_state_dim),
                nn.Linear(self.control_state_dim, 1),
            )
        else:
            self.control_update_mlp = None
            self.control_set_gate = None
            self.control_clear_gate = None
            self.control_input_proj = None
            self.control_router_out = None
            self.control_wait_out = None
            self.control_head = None
        if self.wait_state_dim > 0:
            wait_feature_dim = self.hidden_dim * 2 + self.obs_dim
            self.wait_update_mlp = nn.Sequential(
                nn.LayerNorm(wait_feature_dim),
                nn.Linear(wait_feature_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.wait_state_cell = nn.GRUCell(self.hidden_dim, self.wait_state_dim)
            self.wait_head = nn.Sequential(
                nn.LayerNorm(self.wait_state_dim),
                nn.Linear(self.wait_state_dim, 1),
            )
            self.wait_input_proj = nn.Linear(self.wait_state_dim, self.hidden_dim)
        else:
            self.wait_update_mlp = None
            self.wait_state_cell = None
            self.wait_head = None
            self.wait_input_proj = None
        release_input_dim = self.hidden_dim + self.control_state_dim + self.wait_state_dim
        if self.release_scale != 0.0 or self.release_gate_mode != "bias":
            self.release_head = nn.Sequential(
                nn.LayerNorm(release_input_dim),
                nn.Linear(release_input_dim, 1),
            )
        else:
            self.release_head = None
        self.sink_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.query_readout_proj = None
        self.multiview_baseline_proj = None
        self.multiview_query_proj = None
        self.multiview_fusion = None
        self.multiview_attention = None
        self.multiview_attention_norm = None
        self.multiview_ff = None
        self.multiview_view_dropout = (
            nn.Dropout(self.readout_view_dropout)
            if self.readout_view_dropout > 0.0
            else nn.Identity()
        )
        self.readout_adapter = None
        effective_base_mode = (
            self.readout_mode
            if self.readout_mode not in self.multiview_readout_modes
            else self.readout_base_mode
        )
        if self.readout_mode == "plain":
            readout_input_dim = self.hidden_dim
        elif self.readout_mode == "query_conditioned":
            readout_input_dim = self.hidden_dim * 2
            self.query_readout_proj = nn.Sequential(
                nn.LayerNorm(self.obs_dim),
                nn.Linear(self.obs_dim, self.hidden_dim),
                nn.GELU(),
            )
        elif self.readout_mode == "query_gated":
            readout_input_dim = self.hidden_dim
            self.query_readout_proj = nn.Sequential(
                nn.LayerNorm(self.obs_dim),
                nn.Linear(self.obs_dim, self.hidden_dim),
            )
        elif self.readout_mode == "query_film":
            readout_input_dim = self.hidden_dim
            self.query_readout_proj = nn.Sequential(
                nn.LayerNorm(self.obs_dim),
                nn.Linear(self.obs_dim, self.hidden_dim * 2),
            )
        elif self.readout_mode in self.multiview_readout_modes:
            if effective_base_mode == "query_conditioned":
                self.query_readout_proj = nn.Sequential(
                    nn.LayerNorm(self.obs_dim),
                    nn.Linear(self.obs_dim, self.hidden_dim),
                    nn.GELU(),
                )
                self.multiview_baseline_proj = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.GELU(),
                )
            elif effective_base_mode == "query_gated":
                self.query_readout_proj = nn.Sequential(
                    nn.LayerNorm(self.obs_dim),
                    nn.Linear(self.obs_dim, self.hidden_dim),
                )
            elif effective_base_mode == "query_film":
                self.query_readout_proj = nn.Sequential(
                    nn.LayerNorm(self.obs_dim),
                    nn.Linear(self.obs_dim, self.hidden_dim * 2),
                )
            elif effective_base_mode != "plain":
                raise ValueError(f"Unknown readout_base_mode: {effective_base_mode}")

            if self.readout_mode == "multiview_cross_attention":
                self.multiview_query_proj = nn.Sequential(
                    nn.LayerNorm(self.obs_dim),
                    nn.Linear(self.obs_dim, self.hidden_dim),
                )
                self.multiview_attention = nn.MultiheadAttention(
                    self.hidden_dim,
                    self.readout_attention_heads,
                    batch_first=True,
                )
                self.multiview_attention_norm = nn.LayerNorm(self.hidden_dim)
                self.multiview_ff = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                )
            else:
                self.multiview_fusion = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * len(self.readout_views)),
                    nn.Linear(self.hidden_dim * len(self.readout_views), self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                )
                if self.readout_mode == "multiview_query_gated":
                    self.multiview_query_proj = nn.Sequential(
                        nn.LayerNorm(self.obs_dim),
                        nn.Linear(self.obs_dim, self.hidden_dim),
                    )
                elif self.readout_mode == "multiview_query_film":
                    self.multiview_query_proj = nn.Sequential(
                        nn.LayerNorm(self.obs_dim),
                        nn.Linear(self.obs_dim, self.hidden_dim * 2),
                    )
            readout_input_dim = self.hidden_dim
        else:
            raise ValueError(f"Unknown readout_mode: {self.readout_mode}")

        if self.readout_adapter_mode == "none":
            self.readout_adapter = None
        elif self.readout_adapter_mode == "low_rank":
            rank = max(1, self.readout_adapter_rank)
            self.readout_adapter = LowRankAdapter(self.hidden_dim, rank)
        elif self.readout_adapter_mode == "residual_mlp":
            hidden = max(1, self.readout_adapter_hidden_dim)
            self.readout_adapter = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.hidden_dim),
            )
            nn.init.zeros_(self.readout_adapter[-1].weight)
            nn.init.zeros_(self.readout_adapter[-1].bias)
        elif self.readout_adapter_mode == "affine":
            self.readout_adapter = AffineAdapter(self.hidden_dim)
        else:
            raise ValueError(f"Unknown readout_adapter_mode: {self.readout_adapter_mode}")
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_input_dim),
            nn.Linear(readout_input_dim, self.num_classes),
        )

    def _baseline_readout_input(
        self,
        sink_state: torch.Tensor,
        observations: torch.Tensor,
        *,
        for_multiview: bool = False,
    ) -> torch.Tensor:
        query_obs = observations[:, -1, 0]
        mode = (
            self.readout_mode
            if self.readout_mode not in self.multiview_readout_modes
            else self.readout_base_mode
        )
        if mode == "plain":
            return sink_state
        if mode == "query_conditioned":
            query_context = self.query_readout_proj(query_obs)
            conditioned = torch.cat([sink_state, query_context], dim=-1)
            if for_multiview:
                if self.multiview_baseline_proj is None:
                    raise RuntimeError(
                        "multiview_baseline_proj must be configured for multiview query_conditioned."
                    )
                return self.multiview_baseline_proj(conditioned)
            return conditioned
        if mode == "query_gated":
            query_gate = torch.sigmoid(self.query_readout_proj(query_obs))
            return sink_state * query_gate
        if mode == "query_film":
            film_params = self.query_readout_proj(query_obs)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            return sink_state * (1.0 + torch.tanh(gamma)) + beta
        raise ValueError(f"Unknown baseline readout mode: {mode}")

    def _apply_readout_adapter(self, fused: torch.Tensor) -> torch.Tensor:
        if self.readout_adapter is None or fused.shape[-1] != self.hidden_dim:
            return fused
        return fused + self.readout_adapter(fused)

    def es_parameter_names(self, include_adapters: bool) -> list[str]:
        allowed = [
            "core.router_mlp",
            "core.router_out",
            "core.router_act_out",
            "core.router_wait_out",
            "control_update_mlp",
            "control_set_gate",
            "control_clear_gate",
            "control_router_out",
            "control_wait_out",
            "control_head",
            "wait_update_mlp",
            "wait_state_cell",
            "wait_head",
            "wait_input_proj",
            "release_head",
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
        detach_prefix_steps: int = 0,
        late_window_steps: int = 0,
        forced_actions: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
        oracle_actions: torch.Tensor | None = None,
        oracle_action_mask: torch.Tensor | None = None,
        oracle_route_weight: float = 0.0,
        delay_write_targets: torch.Tensor | None = None,
        delay_write_mask: torch.Tensor | None = None,
        delay_write_weight: float = 0.0,
        memory_payload_targets: torch.Tensor | None = None,
        memory_payload_mask: torch.Tensor | None = None,
        memory_payload_weight: float = 0.0,
        control_targets: torch.Tensor | None = None,
        control_mask: torch.Tensor | None = None,
        control_weight: float = 0.0,
        anti_exit_mask: torch.Tensor | None = None,
        anti_exit_weight: float = 0.0,
        wait_targets: torch.Tensor | None = None,
        wait_mask: torch.Tensor | None = None,
        wait_weight: float = 0.0,
        wait_positive_weight: float = 1.0,
        wait_negative_weight: float = 1.0,
        release_targets: torch.Tensor | None = None,
        release_mask: torch.Tensor | None = None,
        release_weight: float = 0.0,
        release_positive_weight: float = 1.0,
        task_sample_weights: torch.Tensor | None = None,
        return_trace: bool = False,
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
        packet_memory = torch.zeros(
            batch_size,
            self.num_nodes,
            self.packet_memory_slots,
            self.packet_memory_dim,
            device=device,
            dtype=dtype,
        )
        control_state = torch.zeros(
            batch_size,
            self.num_nodes,
            self.control_state_dim,
            device=device,
            dtype=dtype,
        )
        wait_state = torch.zeros(
            batch_size,
            self.num_nodes,
            self.wait_state_dim,
            device=device,
            dtype=dtype,
        )
        sink_state = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        packet_state_query = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

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
        policy_logprob_sum = torch.zeros(batch_size, device=device, dtype=dtype)
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
        memory_read_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        memory_read_weight = torch.zeros(batch_size, device=device, dtype=dtype)
        memory_read_entropy_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        memory_write_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        memory_write_weight = torch.zeros(batch_size, device=device, dtype=dtype)
        memory_write_entropy_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        control_state_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        control_state_weight = torch.zeros(batch_size, device=device, dtype=dtype)
        control_set_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        control_clear_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        control_prob_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        wait_state_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        wait_state_weight = torch.zeros(batch_size, device=device, dtype=dtype)
        wait_prob_sum = torch.zeros(batch_size, device=device, dtype=dtype)

        oracle_route_loss = torch.zeros((), device=device, dtype=dtype)
        oracle_weight_total = torch.zeros((), device=device, dtype=dtype)
        delay_write_loss = torch.zeros((), device=device, dtype=dtype)
        delay_write_total = torch.zeros((), device=device, dtype=dtype)
        memory_payload_loss = torch.zeros((), device=device, dtype=dtype)
        memory_payload_total = torch.zeros((), device=device, dtype=dtype)
        control_loss = torch.zeros((), device=device, dtype=dtype)
        control_total = torch.zeros((), device=device, dtype=dtype)
        anti_exit_loss = torch.zeros((), device=device, dtype=dtype)
        anti_exit_total = torch.zeros((), device=device, dtype=dtype)
        wait_loss = torch.zeros((), device=device, dtype=dtype)
        wait_total = torch.zeros((), device=device, dtype=dtype)
        release_loss = torch.zeros((), device=device, dtype=dtype)
        release_total = torch.zeros((), device=device, dtype=dtype)
        release_prob_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        release_prob_weight = torch.zeros(batch_size, device=device, dtype=dtype)

        trace: dict[str, torch.Tensor] | None = None
        if return_trace:
            trace = {
                "active_mass": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "alive_mass": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "packet_age": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "mailbox_mass": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "action_mass": torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype),
                "router_logits": torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype),
                "router_probs": torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype),
                "packet_state": torch.zeros(batch_size, seq_len, self.hidden_dim, device=device, dtype=dtype),
                "sink_state": torch.zeros(batch_size, seq_len, self.hidden_dim, device=device, dtype=dtype),
                "memory_read_gate": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "memory_write_gate": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "memory_read_entropy": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "memory_write_entropy": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
                "final_sink_state": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
                "sink_state_query": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
                "packet_state_query": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
                "baseline_readout_input": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            }
            if self.packet_memory_slots > 0:
                trace["memory_read_state"] = torch.zeros(
                    batch_size,
                    seq_len,
                    self.packet_memory_dim,
                    device=device,
                    dtype=dtype,
                )
                trace["memory_read_weights"] = torch.zeros(
                    batch_size,
                    seq_len,
                    self.packet_memory_slots,
                    device=device,
                    dtype=dtype,
                )
                trace["memory_write_weights"] = torch.zeros(
                    batch_size,
                    seq_len,
                    self.packet_memory_slots,
                    device=device,
                    dtype=dtype,
                )
            if self.control_state_dim > 0:
                trace["control_state"] = torch.zeros(
                    batch_size,
                    seq_len,
                    self.control_state_dim,
                    device=device,
                    dtype=dtype,
                )
                trace["control_prob"] = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
            if self.wait_state_dim > 0:
                trace["wait_state"] = torch.zeros(
                    batch_size,
                    seq_len,
                    self.wait_state_dim,
                    device=device,
                    dtype=dtype,
                )
            if self.routing_head_mode != "flat" or self.wait_state_dim > 0:
                trace["wait_prob"] = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
            if self.release_head is not None:
                trace["release_prob"] = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        late_window_boundary = seq_len - late_window_steps if late_window_steps > 0 else -1
        prefix_detached = False
        late_window_detached = False
        for time_index in range(seq_len):
            obs_t = observations[:, time_index]
            carry_mass = torch.zeros_like(packet_masses)
            carry_state_sum = torch.zeros_like(packet_states)
            carry_age_sum = torch.zeros_like(packet_ages)
            carry_memory_sum = torch.zeros_like(packet_memory)
            carry_control_sum = torch.zeros_like(control_state)
            carry_wait_sum = torch.zeros_like(wait_state)

            if trace is not None:
                time_active_mass = torch.zeros(batch_size, device=device, dtype=dtype)
                time_router_mass = torch.zeros(batch_size, device=device, dtype=dtype)
                time_action_sum = torch.zeros(batch_size, 3, device=device, dtype=dtype)
                time_router_logits_sum = torch.zeros(batch_size, 3, device=device, dtype=dtype)
                time_router_probs_sum = torch.zeros(batch_size, 3, device=device, dtype=dtype)
                time_packet_state_sum = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
                time_age_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                time_memory_read_gate_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                time_memory_write_gate_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                time_memory_read_entropy_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                time_memory_write_entropy_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                if self.packet_memory_slots > 0:
                    time_memory_read_state_sum = torch.zeros(
                        batch_size,
                        self.packet_memory_dim,
                        device=device,
                        dtype=dtype,
                    )
                    time_memory_read_weights_sum = torch.zeros(
                        batch_size,
                        self.packet_memory_slots,
                        device=device,
                        dtype=dtype,
                    )
                    time_memory_write_weights_sum = torch.zeros(
                        batch_size,
                        self.packet_memory_slots,
                        device=device,
                        dtype=dtype,
                    )
                if self.control_state_dim > 0:
                    time_control_state_sum = torch.zeros(
                        batch_size,
                        self.control_state_dim,
                        device=device,
                        dtype=dtype,
                    )
                    time_control_prob_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                if self.wait_state_dim > 0:
                    time_wait_state_sum = torch.zeros(
                        batch_size,
                        self.wait_state_dim,
                        device=device,
                        dtype=dtype,
                    )
                if self.routing_head_mode != "flat" or self.wait_state_dim > 0:
                    time_wait_prob_sum = torch.zeros(batch_size, device=device, dtype=dtype)
                if self.release_head is not None:
                    time_release_prob_sum = torch.zeros(batch_size, device=device, dtype=dtype)

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
                active_mass = mass.sum(dim=1)
                age_values = packet_ages.squeeze(-1)

                packet_core_input = packet_states
                memory_write_gate = None
                updated_memory = packet_memory
                memory_read = None
                memory_read_gate = None
                memory_read_weights = None
                memory_write_weights = None
                if self.packet_memory_slots > 0:
                    memory_read, memory_read_gate, memory_read_weights = self._read_packet_memory(
                        packet_memory=packet_memory,
                        packet_state=packet_states,
                        node_state=node_states,
                        observations=obs_t,
                    )
                    memory_read_entropy = -(memory_read_weights * torch.log(memory_read_weights.clamp_min(1e-8))).sum(dim=-1)
                    memory_read_sum = memory_read_sum + (memory_read_gate.squeeze(-1) * mass).sum(dim=1)
                    memory_read_weight = memory_read_weight + active_mass
                    memory_read_entropy_sum = memory_read_entropy_sum + (memory_read_entropy * mass).sum(dim=1)
                    packet_core_input = packet_states + memory_read_gate * self.memory_input_proj(memory_read)
                    if (
                        memory_payload_targets is not None
                        and memory_payload_mask is not None
                        and memory_payload_weight > 0.0
                    ):
                        payload_target = memory_payload_targets[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        payload_mask = memory_payload_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        payload_logits = self.memory_payload_head(memory_read.float())
                        per_node_ce = F.cross_entropy(
                            payload_logits.reshape(-1, self.num_classes),
                            payload_target.reshape(-1),
                            reduction="none",
                        ).reshape(batch_size, self.num_nodes).to(dtype)
                        weight = payload_mask * mass
                        memory_payload_loss = memory_payload_loss + (per_node_ce * weight).sum()
                        memory_payload_total = memory_payload_total + weight.sum()
                if self.control_state_dim > 0:
                    packet_core_input = packet_core_input + (
                        self.control_input_scale * self.control_input_proj(control_state)
                    )
                if self.wait_state_dim > 0 and self.wait_input_proj is not None:
                    packet_core_input = packet_core_input + (
                        self.wait_state_input_scale * self.wait_input_proj(wait_state)
                    )

                if trace is not None:
                    time_active_mass = time_active_mass + active_mass
                    time_packet_state_sum = time_packet_state_sum + (
                        packet_core_input * mass.unsqueeze(-1)
                    ).sum(dim=1)
                    time_age_sum = time_age_sum + (age_values * mass).sum(dim=1)
                    if self.packet_memory_slots > 0 and memory_read is not None and memory_read_gate is not None and memory_read_weights is not None:
                        time_memory_read_gate_sum = time_memory_read_gate_sum + (
                            memory_read_gate.squeeze(-1) * mass
                        ).sum(dim=1)
                        time_memory_read_state_sum = time_memory_read_state_sum + (
                            memory_read * mass.unsqueeze(-1)
                        ).sum(dim=1)
                        time_memory_read_weights_sum = time_memory_read_weights_sum + (
                            memory_read_weights * mass.unsqueeze(-1)
                        ).sum(dim=1)
                    if self.control_state_dim > 0:
                        time_control_state_sum = time_control_state_sum + (
                            control_state * mass.unsqueeze(-1)
                        ).sum(dim=1)
                        control_logits_live = self.control_head(control_state.float()).squeeze(-1).to(dtype)
                        control_prob_live = torch.sigmoid(control_logits_live)
                        time_control_prob_sum = time_control_prob_sum + (control_prob_live * mass).sum(dim=1)
                    if self.wait_state_dim > 0:
                        time_wait_state_sum = time_wait_state_sum + (
                            wait_state * mass.unsqueeze(-1)
                        ).sum(dim=1)

                core_output = self.core(
                    packet_state=packet_core_input,
                    node_state=node_states,
                    observations=obs_t,
                    node_index=node_indices.expand(batch_size, -1),
                    age_fraction=age_fraction,
                    time_fraction=time_fraction,
                    remaining_fraction=remaining_fraction,
                )
                if len(core_output) == 5:
                    node_state_next, packet_next, logits, delay_retain, router_aux = core_output
                elif len(core_output) == 4:
                    node_state_next, packet_next, logits, delay_retain = core_output
                    router_aux = {"act_logits": None, "wait_logit": None}
                elif len(core_output) == 3:
                    node_state_next, packet_next, logits = core_output
                    delay_retain = None
                    router_aux = {"act_logits": None, "wait_logit": None}
                else:
                    raise ValueError(f"Unexpected core output size: {len(core_output)}")

                if self.packet_memory_slots > 0:
                    updated_memory, memory_write_gate, memory_write_weights = self._write_packet_memory(
                        packet_memory=packet_memory,
                        packet_state=packet_next,
                        node_state=node_state_next,
                        observations=obs_t,
                    )
                    memory_write_entropy = -(memory_write_weights * torch.log(memory_write_weights.clamp_min(1e-8))).sum(dim=-1)
                    memory_write_sum = memory_write_sum + (memory_write_gate.squeeze(-1) * mass).sum(dim=1)
                    memory_write_weight = memory_write_weight + active_mass
                    memory_write_entropy_sum = memory_write_entropy_sum + (memory_write_entropy * mass).sum(dim=1)

                updated_control = control_state
                control_set = None
                control_clear = None
                if self.control_state_dim > 0:
                    updated_control, control_set, control_clear = self._update_control_state(
                        control_state=control_state,
                        packet_state=packet_next,
                        node_state=node_state_next,
                        observations=obs_t,
                    )
                    control_state_mean = updated_control.mean(dim=-1)
                    control_state_sum = control_state_sum + (control_state_mean * mass).sum(dim=1)
                    control_state_weight = control_state_weight + active_mass
                    if control_set is not None:
                        control_set_sum = control_set_sum + (control_set.mean(dim=-1) * mass).sum(dim=1)
                    if control_clear is not None:
                        control_clear_sum = control_clear_sum + (control_clear.mean(dim=-1) * mass).sum(dim=1)
                    control_logits = self.control_head(updated_control.float()).squeeze(-1).to(dtype)
                    control_probs = torch.sigmoid(control_logits)
                    control_prob_sum = control_prob_sum + (control_probs * mass).sum(dim=1)
                    if (
                        control_targets is not None
                        and control_mask is not None
                        and control_weight > 0.0
                    ):
                        target = control_targets[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        target_mask = control_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        per_node_bce = F.binary_cross_entropy_with_logits(
                            control_logits.float(),
                            target.float(),
                            reduction="none",
                        ).to(dtype)
                        weight = target_mask * mass
                        control_loss = control_loss + (per_node_bce * weight).sum()
                        control_total = control_total + weight.sum()

                updated_wait = wait_state
                if self.wait_state_dim > 0:
                    updated_wait = self._update_wait_state(
                        wait_state=wait_state,
                        packet_state=packet_next,
                        node_state=node_state_next,
                        observations=obs_t,
                    )
                    wait_state_mean = updated_wait.mean(dim=-1)
                    wait_state_sum = wait_state_sum + (wait_state_mean * mass).sum(dim=1)
                    wait_state_weight = wait_state_weight + active_mass

                wait_logit = self._select_wait_logit(
                    core_wait_logit=router_aux.get("wait_logit"),
                    updated_control=updated_control,
                    updated_wait=updated_wait,
                    router_hidden=router_aux.get("router_hidden"),
                )
                release_logit = self._compute_release_logit(
                    router_hidden=router_aux.get("router_hidden"),
                    updated_control=updated_control,
                    updated_wait=updated_wait,
                )
                if release_logit is not None:
                    release_probs = torch.sigmoid(release_logit.float()).to(dtype).squeeze(-1)
                    release_prob_sum = release_prob_sum + (release_probs * mass).sum(dim=1)
                    release_prob_weight = release_prob_weight + active_mass
                    if trace is not None:
                        time_release_prob_sum = time_release_prob_sum + (release_probs * mass).sum(dim=1)
                    if (
                        release_targets is not None
                        and release_mask is not None
                        and release_weight > 0.0
                    ):
                        target = release_targets[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        target_mask = release_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        positive_scale = torch.where(
                            target > 0.5,
                            torch.full_like(target, float(release_positive_weight)),
                            torch.ones_like(target),
                        )
                        per_node_bce = F.binary_cross_entropy_with_logits(
                            release_logit.float().squeeze(-1),
                            target.float(),
                            reduction="none",
                        ).to(dtype)
                        weight = target_mask * mass * positive_scale
                        release_loss = release_loss + (per_node_bce * weight).sum()
                        release_total = release_total + weight.sum()
                if wait_logit is not None:
                    wait_probs = torch.sigmoid(wait_logit.float()).to(dtype).squeeze(-1)
                    wait_prob_sum = wait_prob_sum + (wait_probs * mass).sum(dim=1)
                    if wait_targets is not None and wait_mask is not None and wait_weight > 0.0:
                        target = wait_targets[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        target_mask = wait_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                        class_scale = torch.where(
                            target > 0.5,
                            torch.full_like(target, float(wait_positive_weight)),
                            torch.full_like(target, float(wait_negative_weight)),
                        )
                        per_node_bce = F.binary_cross_entropy_with_logits(
                            wait_logit.float().squeeze(-1),
                            target.float(),
                            reduction="none",
                        ).to(dtype)
                        weight = target_mask * mass * class_scale
                        wait_loss = wait_loss + (per_node_bce * weight).sum()
                        wait_total = wait_total + weight.sum()

                masked_logits = self._compose_routing_logits(
                    flat_logits=logits,
                    act_logits=router_aux.get("act_logits"),
                    wait_logit=wait_logit,
                    release_logit=release_logit,
                    control_state=control_state,
                )
                if action_masks is not None:
                    masked_logits = self._apply_action_mask(masked_logits, action_masks[:, time_index])

                router_probs = F.softmax(masked_logits / max(temperature, 1e-6), dim=-1)
                router_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1)
                router_conf = router_probs.max(dim=-1).values
                route_entropy_sum = route_entropy_sum + (router_entropy * mass).sum(dim=1)
                route_conf_sum = route_conf_sum + (router_conf * mass).sum(dim=1)
                route_weight_total = route_weight_total + active_mass

                if trace is not None:
                    time_router_mass = time_router_mass + active_mass
                    time_router_logits_sum = time_router_logits_sum + (masked_logits * mass.unsqueeze(-1)).sum(dim=1)
                    time_router_probs_sum = time_router_probs_sum + (router_probs * mass.unsqueeze(-1)).sum(dim=1)
                    if wait_logit is not None:
                        time_wait_prob_sum = time_wait_prob_sum + (
                            torch.sigmoid(wait_logit.float()).to(dtype).squeeze(-1) * mass
                        ).sum(dim=1)
                    if self.packet_memory_slots > 0 and memory_write_gate is not None and memory_write_weights is not None:
                        time_memory_write_gate_sum = time_memory_write_gate_sum + (
                            memory_write_gate.squeeze(-1) * mass
                        ).sum(dim=1)
                        time_memory_write_weights_sum = time_memory_write_weights_sum + (
                            memory_write_weights * mass.unsqueeze(-1)
                        ).sum(dim=1)
                    if self.packet_memory_slots > 0 and memory_read_weights is not None:
                        time_memory_read_entropy_sum = time_memory_read_entropy_sum + (
                            memory_read_entropy * mass
                        ).sum(dim=1)
                    if self.packet_memory_slots > 0 and memory_write_weights is not None:
                        time_memory_write_entropy_sum = time_memory_write_entropy_sum + (
                            memory_write_entropy * mass
                        ).sum(dim=1)

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
                if delay_write_targets is not None and delay_write_mask is not None and delay_write_weight > 0.0:
                    write_target = delay_write_targets[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    write_mask = delay_write_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    write_prob: torch.Tensor | None = None
                    if memory_write_gate is not None:
                        write_prob = memory_write_gate.squeeze(-1).float().clamp(1e-4, 1.0 - 1e-4)
                    elif delay_retain is not None:
                        write_prob = (1.0 - delay_retain.squeeze(-1).float()).clamp(1e-4, 1.0 - 1e-4)
                    if write_prob is not None:
                        write_logits = torch.logit(write_prob)
                        per_node_bce = F.binary_cross_entropy_with_logits(
                            write_logits,
                            write_target.float(),
                            reduction="none",
                        ).to(dtype)
                        weight = (write_mask * mass).float()
                        delay_write_loss = delay_write_loss + (per_node_bce * weight).sum()
                        delay_write_total = delay_write_total + weight.sum()
                if anti_exit_mask is not None and anti_exit_weight > 0.0:
                    anti_exit_step_mask = anti_exit_mask[:, time_index].unsqueeze(1).expand(-1, self.num_nodes)
                    exit_prob = router_probs[..., ACTION_EXIT].float().clamp_max(1.0 - 1e-6)
                    penalty = -torch.log1p(-exit_prob)
                    weight = (anti_exit_step_mask * mass).float()
                    anti_exit_loss = anti_exit_loss + (penalty.to(dtype) * weight.to(dtype)).sum()
                    anti_exit_total = anti_exit_total + weight.sum().to(dtype)

                selected_logprob = None
                if route_mode == "sample":
                    sampled_indices = torch.multinomial(
                        router_probs.reshape(-1, router_probs.shape[-1]),
                        num_samples=1,
                    ).reshape(batch_size, self.num_nodes)
                    actions = F.one_hot(sampled_indices, num_classes=3).to(dtype)
                    selected_logprob = torch.log(
                        torch.gather(
                            router_probs,
                            dim=-1,
                            index=sampled_indices.unsqueeze(-1),
                        ).squeeze(-1).clamp_min(1e-8)
                    )
                else:
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
                        if selected_logprob is not None:
                            selected_logprob = torch.where(
                                valid.view(batch_size, 1),
                                torch.zeros_like(selected_logprob),
                                selected_logprob,
                            )
                if selected_logprob is not None:
                    policy_logprob_sum = policy_logprob_sum + (selected_logprob.to(dtype) * mass).sum(dim=1)

                action_summary = (mass.unsqueeze(-1) * actions).sum(dim=1)
                if trace is not None:
                    time_action_sum = time_action_sum + action_summary.detach()
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
                carry_memory_sum = carry_memory_sum + delay_mass.unsqueeze(-1).unsqueeze(-1) * updated_memory
                carry_control_sum = carry_control_sum + delay_mass.unsqueeze(-1) * updated_control
                carry_wait_sum = carry_wait_sum + delay_mass.unsqueeze(-1) * updated_wait
                if delay_retain is not None:
                    delay_retain_sum = delay_retain_sum + (delay_retain.squeeze(-1) * delay_mass).sum(dim=1)
                    delay_retain_weight = delay_retain_weight + delay_mass.sum(dim=1)

                next_mass = torch.zeros_like(packet_masses)
                next_state_sum = torch.zeros_like(packet_states)
                next_age_sum = torch.zeros_like(packet_ages)
                next_memory_sum = torch.zeros_like(packet_memory)
                next_control_sum = torch.zeros_like(control_state)
                next_wait_sum = torch.zeros_like(wait_state)
                if self.num_nodes > 1:
                    next_mass[:, 1:] = next_mass[:, 1:] + forward_mass[:, :-1]
                    next_state_sum[:, 1:, :] = next_state_sum[:, 1:, :] + weighted_packet[:, :-1, :] * actions[:, :-1, ACTION_FORWARD].unsqueeze(-1)
                    next_age_sum[:, 1:, :] = next_age_sum[:, 1:, :] + next_age[:, :-1, :] * forward_mass[:, :-1].unsqueeze(-1)
                    next_memory_sum[:, 1:, :, :] = next_memory_sum[:, 1:, :, :] + (
                        updated_memory[:, :-1, :, :]
                        * forward_mass[:, :-1].unsqueeze(-1).unsqueeze(-1)
                    )
                    next_control_sum[:, 1:, :] = next_control_sum[:, 1:, :] + (
                        updated_control[:, :-1, :] * forward_mass[:, :-1].unsqueeze(-1)
                    )
                    next_wait_sum[:, 1:, :] = next_wait_sum[:, 1:, :] + (
                        updated_wait[:, :-1, :] * forward_mass[:, :-1].unsqueeze(-1)
                    )

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
                packet_memory = self._normalize_weighted(next_memory_sum, next_mass, eps)
                control_state = self._normalize_weighted(next_control_sum, next_mass, eps)
                wait_state = self._normalize_weighted(next_wait_sum, next_mass, eps)

            forced_delay = packet_masses
            if float(forced_delay.max().detach().cpu()) > 0.0:
                delays = delays + forced_delay.sum(dim=1)
                carry_mass = carry_mass + forced_delay
                carry_state_sum = carry_state_sum + forced_delay.unsqueeze(-1) * packet_states
                carry_age_sum = carry_age_sum + forced_delay.unsqueeze(-1) * (packet_ages + 1.0)
                carry_memory_sum = carry_memory_sum + forced_delay.unsqueeze(-1).unsqueeze(-1) * packet_memory
                carry_control_sum = carry_control_sum + forced_delay.unsqueeze(-1) * control_state
                carry_wait_sum = carry_wait_sum + forced_delay.unsqueeze(-1) * wait_state

            packet_masses = carry_mass
            packet_states = self._normalize_state(carry_state_sum, carry_mass, eps)
            packet_ages = self._normalize_state(carry_age_sum, carry_mass, eps)
            packet_memory = self._normalize_weighted(carry_memory_sum, carry_mass, eps)
            control_state = self._normalize_weighted(carry_control_sum, carry_mass, eps)
            wait_state = self._normalize_weighted(carry_wait_sum, carry_mass, eps)
            mailbox_load = carry_mass.sum(dim=1)
            mailbox_mass_sum = mailbox_mass_sum + mailbox_load
            mailbox_peak = torch.maximum(mailbox_peak, mailbox_load)
            mailbox_steps = mailbox_steps + (mailbox_load >= 0.0).float()
            packet_state_query = (packet_masses.unsqueeze(-1) * packet_states).sum(dim=1) / packet_masses.sum(dim=1).clamp_min(float(eps)).unsqueeze(-1)

            if trace is not None:
                trace["active_mass"][:, time_index] = time_active_mass
                trace["alive_mass"][:, time_index] = packet_masses.sum(dim=1)
                trace["packet_age"][:, time_index] = time_age_sum / time_active_mass.clamp_min(float(eps))
                trace["mailbox_mass"][:, time_index] = mailbox_load
                trace["action_mass"][:, time_index, :] = time_action_sum
                trace["router_logits"][:, time_index, :] = time_router_logits_sum / time_router_mass.clamp_min(float(eps)).unsqueeze(-1)
                trace["router_probs"][:, time_index, :] = time_router_probs_sum / time_router_mass.clamp_min(float(eps)).unsqueeze(-1)
                trace["packet_state"][:, time_index, :] = packet_state_query
                trace["sink_state"][:, time_index, :] = sink_state
                trace["memory_read_gate"][:, time_index] = time_memory_read_gate_sum / time_active_mass.clamp_min(float(eps))
                trace["memory_write_gate"][:, time_index] = time_memory_write_gate_sum / time_active_mass.clamp_min(float(eps))
                trace["memory_read_entropy"][:, time_index] = time_memory_read_entropy_sum / time_active_mass.clamp_min(float(eps))
                trace["memory_write_entropy"][:, time_index] = time_memory_write_entropy_sum / time_active_mass.clamp_min(float(eps))
                if self.packet_memory_slots > 0:
                    trace["memory_read_state"][:, time_index, :] = time_memory_read_state_sum / time_active_mass.clamp_min(float(eps)).unsqueeze(-1)
                    trace["memory_read_weights"][:, time_index, :] = time_memory_read_weights_sum / time_active_mass.clamp_min(float(eps)).unsqueeze(-1)
                    trace["memory_write_weights"][:, time_index, :] = time_memory_write_weights_sum / time_active_mass.clamp_min(float(eps)).unsqueeze(-1)
                if self.control_state_dim > 0:
                    trace["control_state"][:, time_index, :] = time_control_state_sum / time_active_mass.clamp_min(float(eps)).unsqueeze(-1)
                    trace["control_prob"][:, time_index] = time_control_prob_sum / time_active_mass.clamp_min(float(eps))
                if self.wait_state_dim > 0:
                    trace["wait_state"][:, time_index, :] = time_wait_state_sum / time_active_mass.clamp_min(float(eps)).unsqueeze(-1)
                if "wait_prob" in trace:
                    trace["wait_prob"][:, time_index] = time_wait_prob_sum / time_active_mass.clamp_min(float(eps))
                if "release_prob" in trace:
                    trace["release_prob"][:, time_index] = time_release_prob_sum / time_active_mass.clamp_min(float(eps))

            should_detach = False
            if truncate_bptt_steps > 0 and (time_index + 1) % truncate_bptt_steps == 0:
                should_detach = True
            if detach_prefix_steps > 0 and not prefix_detached and (time_index + 1) >= detach_prefix_steps:
                should_detach = True
                prefix_detached = True
            if late_window_steps > 0 and not late_window_detached and (time_index + 1) >= late_window_boundary:
                should_detach = True
                late_window_detached = True
            if should_detach:
                node_states = node_states.detach()
                packet_states = packet_states.detach()
                packet_masses = packet_masses.detach()
                packet_ages = packet_ages.detach()
                packet_memory = packet_memory.detach()
                control_state = control_state.detach()
                wait_state = wait_state.detach()
                sink_state = sink_state.detach()

        sink_state_query = sink_state
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
        if trace is not None:
            trace["final_sink_state"] = sink_state

        if self.readout_mode in self.multiview_readout_modes:
            baseline_view = self._baseline_readout_input(
                sink_state,
                observations,
                for_multiview=True,
            )
            view_map = {
                "final_sink_state": sink_state,
                "sink_state_query": sink_state_query,
                "packet_state_query": packet_state_query,
                "baseline_readout_input": baseline_view,
            }
            view_tensors = [view_map[name] for name in self.readout_views]
            if self.readout_mode == "multiview_cross_attention":
                view_bank = self.multiview_view_dropout(torch.stack(view_tensors, dim=1))
                latent = self.multiview_query_proj(observations[:, -1, 0])
                for _ in range(max(1, self.readout_iter_steps)):
                    attended, _ = self.multiview_attention(
                        latent.unsqueeze(1),
                        view_bank,
                        view_bank,
                        need_weights=False,
                    )
                    latent = self.multiview_attention_norm(latent + attended.squeeze(1))
                    latent = latent + self.multiview_ff(latent)
                readout_input = latent
            else:
                fused = self.multiview_fusion(
                    torch.cat(
                        [self.multiview_view_dropout(view) for view in view_tensors],
                        dim=-1,
                    )
                )
                if self.readout_mode == "multiview_query_gated":
                    query_gate = torch.sigmoid(self.multiview_query_proj(observations[:, -1, 0]))
                    fused = fused * query_gate
                elif self.readout_mode == "multiview_query_film":
                    film_params = self.multiview_query_proj(observations[:, -1, 0])
                    gamma, beta = torch.chunk(film_params, 2, dim=-1)
                    fused = fused * (1.0 + torch.tanh(gamma)) + beta
                readout_input = fused
            readout_input = self._apply_readout_adapter(readout_input)
        else:
            baseline_view = self._baseline_readout_input(
                sink_state,
                observations,
                for_multiview=False,
            )
            readout_input = baseline_view
        if self.readout_mode not in self.multiview_readout_modes:
            readout_input = self._apply_readout_adapter(readout_input)
        if trace is not None:
            trace["sink_state_query"] = sink_state_query
            trace["packet_state_query"] = packet_state_query
            if baseline_view.shape[-1] == self.hidden_dim:
                trace["baseline_readout_input"] = baseline_view
            trace["final_readout_input"] = readout_input
        logits = self.readout(readout_input)
        if task_sample_weights is None:
            task_loss = F.cross_entropy(logits, labels)
        else:
            per_sample_task_loss = F.cross_entropy(logits, labels, reduction="none")
            weights = task_sample_weights.to(device=device, dtype=per_sample_task_loss.dtype)
            task_loss = (per_sample_task_loss * weights).sum() / weights.sum().clamp_min(float(eps))
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
        memory_payload_term = torch.zeros((), device=device, dtype=dtype)
        if memory_payload_weight > 0.0 and float(memory_payload_total.detach().cpu()) > 0.0:
            memory_payload_term = memory_payload_weight * (
                memory_payload_loss / memory_payload_total.clamp_min(float(eps))
            )
        control_term = torch.zeros((), device=device, dtype=dtype)
        if control_weight > 0.0 and float(control_total.detach().cpu()) > 0.0:
            control_term = control_weight * (control_loss / control_total.clamp_min(float(eps)))
        anti_exit_term = torch.zeros((), device=device, dtype=dtype)
        if anti_exit_weight > 0.0 and float(anti_exit_total.detach().cpu()) > 0.0:
            anti_exit_term = anti_exit_weight * (anti_exit_loss / anti_exit_total.clamp_min(float(eps)))
        wait_term = torch.zeros((), device=device, dtype=dtype)
        if wait_weight > 0.0 and float(wait_total.detach().cpu()) > 0.0:
            wait_term = wait_weight * (wait_loss / wait_total.clamp_min(float(eps)))
        release_term = torch.zeros((), device=device, dtype=dtype)
        if release_weight > 0.0 and float(release_total.detach().cpu()) > 0.0:
            release_term = release_weight * (release_loss / release_total.clamp_min(float(eps)))
        total_loss = task_loss + route_penalty + oracle_route_term + delay_write_term + memory_payload_term + control_term + anti_exit_term + wait_term + release_term

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

        delay_write_mean = (
            memory_write_sum / memory_write_weight.clamp_min(float(eps))
            if self.packet_memory_slots > 0
            else 1.0 - (delay_retain_sum / delay_retain_weight.clamp_min(float(eps)))
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
            "forward_rate": action_rates[:, ACTION_FORWARD],
            "exit_rate": action_rates[:, ACTION_EXIT],
            "delay_rate": action_rates[:, ACTION_DELAY],
            "route_entropy": route_entropy_sum / route_weight_total.clamp_min(float(eps)),
            "router_confidence": route_conf_sum / route_weight_total.clamp_min(float(eps)),
            "policy_logprob": policy_logprob_sum,
            "mailbox_occupancy": mailbox_mass_sum / mailbox_steps.clamp_min(1.0),
            "mailbox_peak": mailbox_peak,
            "packet_age_mean": age_mass_sum / age_weight_total.clamp_min(float(eps)),
            "packet_age_max": max_age,
            "exit_age_mean": exit_age_sum / exit_age_mass.clamp_min(float(eps)),
            "decision_count": decision_mass_total,
            "delay_retain_mean": delay_retain_sum / delay_retain_weight.clamp_min(float(eps)),
            "delay_write_mean": delay_write_mean,
            "memory_read_mean": memory_read_sum / memory_read_weight.clamp_min(float(eps)),
            "memory_write_mean": memory_write_sum / memory_write_weight.clamp_min(float(eps)),
            "memory_read_entropy": memory_read_entropy_sum / memory_read_weight.clamp_min(float(eps)),
            "memory_write_entropy": memory_write_entropy_sum / memory_write_weight.clamp_min(float(eps)),
            "control_state_mean": control_state_sum / control_state_weight.clamp_min(float(eps)),
            "control_set_mean": control_set_sum / control_state_weight.clamp_min(float(eps)),
            "control_clear_mean": control_clear_sum / control_state_weight.clamp_min(float(eps)),
            "control_prob_mean": control_prob_sum / control_state_weight.clamp_min(float(eps)),
            "wait_state_mean": wait_state_sum / wait_state_weight.clamp_min(float(eps)),
            "wait_prob_mean": wait_prob_sum / route_weight_total.clamp_min(float(eps)),
            "release_prob_mean": release_prob_sum / release_prob_weight.clamp_min(float(eps)),
            "penalty_hops": torch.full_like(accuracy, float(penalty_hops.detach().item())),
            "penalty_delays": torch.full_like(accuracy, float(penalty_delays.detach().item())),
            "penalty_ttl": torch.full_like(accuracy, float(penalty_ttl.detach().item())),
            "oracle_route_loss": torch.full_like(accuracy, float(oracle_route_term.detach().item())),
            "delay_write_loss": torch.full_like(accuracy, float(delay_write_term.detach().item())),
            "memory_payload_loss": torch.full_like(accuracy, float(memory_payload_term.detach().item())),
            "control_loss": torch.full_like(accuracy, float(control_term.detach().item())),
            "anti_exit_loss": torch.full_like(accuracy, float(anti_exit_term.detach().item())),
            "wait_loss": torch.full_like(accuracy, float(wait_term.detach().item())),
            "release_loss": torch.full_like(accuracy, float(release_term.detach().item())),
        }
        for src_index, src_name in ACTION_NAMES.items():
            for dst_index, dst_name in ACTION_NAMES.items():
                stats[f"transition_{src_name}_to_{dst_name}"] = transition_norm[:, src_index, dst_index]

        return RoutingForwardOutput(
            logits=logits,
            sink_state=sink_state,
            loss=total_loss,
            route_loss=route_penalty + oracle_route_term + delay_write_term + memory_payload_term + control_term + anti_exit_term + wait_term + release_term,
            task_loss=task_loss,
            stats=stats,
            trace=trace,
        )

    @staticmethod
    def _normalize_state(
        weighted_state: torch.Tensor,
        mass: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        return PacketRoutingModel._normalize_weighted(weighted_state, mass, eps)

    @staticmethod
    def _normalize_weighted(
        weighted_tensor: torch.Tensor,
        mass: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        denom = mass
        mask = mass
        while denom.ndim < weighted_tensor.ndim:
            denom = denom.unsqueeze(-1)
            mask = mask.unsqueeze(-1)
        normalized = weighted_tensor / denom.clamp_min(float(eps))
        return torch.where(mask > 0.0, normalized, torch.zeros_like(weighted_tensor))

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

    def _read_packet_memory(
        self,
        packet_memory: torch.Tensor,
        packet_state: torch.Tensor,
        node_state: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.packet_memory_slots <= 0:
            raise ValueError("Packet memory is disabled.")
        memory_features = torch.cat([packet_state, node_state, observations], dim=-1)
        read_hidden = self.memory_read_mlp(memory_features)
        read_weights = F.softmax(self.memory_read_slots(read_hidden), dim=-1)
        read_gate = torch.sigmoid(self.memory_read_gate(read_hidden))
        read_state = (packet_memory * read_weights.unsqueeze(-1)).sum(dim=-2)
        return read_state, read_gate, read_weights

    def _write_packet_memory(
        self,
        packet_memory: torch.Tensor,
        packet_state: torch.Tensor,
        node_state: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.packet_memory_slots <= 0:
            raise ValueError("Packet memory is disabled.")
        memory_features = torch.cat([packet_state, node_state, observations], dim=-1)
        write_hidden = self.memory_write_mlp(memory_features)
        write_weights = F.softmax(self.memory_write_slots(write_hidden), dim=-1)
        write_gate = torch.sigmoid(self.memory_write_gate(write_hidden))
        write_value = self.memory_write_value(write_hidden)
        slot_gate = write_gate * write_weights
        updated_memory = packet_memory + slot_gate.unsqueeze(-1) * (
            write_value.unsqueeze(-2) - packet_memory
        )
        return updated_memory, write_gate, write_weights

    def _update_control_state(
        self,
        control_state: torch.Tensor,
        packet_state: torch.Tensor,
        node_state: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.control_state_dim <= 0:
            raise ValueError("Control state is disabled.")
        control_features = torch.cat([packet_state, node_state, observations], dim=-1)
        control_hidden = self.control_update_mlp(control_features)
        control_set = torch.sigmoid(self.control_set_gate(control_hidden))
        if self.control_state_mode == "set_clear" and self.control_clear_gate is not None:
            control_clear = torch.sigmoid(self.control_clear_gate(control_hidden))
            updated_control = control_state * (1.0 - control_clear) + (1.0 - control_state) * control_set
            return updated_control.clamp_(0.0, 1.0), control_set, control_clear
        updated_control = control_state + (1.0 - control_state) * control_set
        return updated_control.clamp_(0.0, 1.0), control_set, None

    def _update_wait_state(
        self,
        wait_state: torch.Tensor,
        packet_state: torch.Tensor,
        node_state: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if self.wait_state_dim <= 0 or self.wait_update_mlp is None or self.wait_state_cell is None:
            raise ValueError("Wait state is disabled.")
        wait_features = torch.cat([packet_state, node_state, observations], dim=-1)
        wait_hidden = self.wait_update_mlp(wait_features)
        batch_shape = wait_state.shape[:-1]
        return self.wait_state_cell(
            wait_hidden.reshape(-1, self.hidden_dim),
            wait_state.reshape(-1, self.wait_state_dim),
        ).reshape(*batch_shape, self.wait_state_dim)

    def _select_wait_logit(
        self,
        *,
        core_wait_logit: torch.Tensor | None,
        updated_control: torch.Tensor,
        updated_wait: torch.Tensor,
        router_hidden: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.routing_head_mode == "flat":
            return None
        if self.routing_head_mode == "control_wait_act":
            if self.control_head is None:
                raise ValueError("control_wait_act requires control_state_dim > 0.")
            return self.control_head(updated_control.float()).to(updated_control.dtype)
        if self.routing_head_mode == "recurrent_wait_act":
            if self.wait_head is None:
                raise ValueError("recurrent_wait_act requires wait_state_dim > 0.")
            return self.wait_head(updated_wait.float()).to(updated_wait.dtype)
        if core_wait_logit is None:
            raise ValueError(f"{self.routing_head_mode} requires core wait logits.")
        wait_logit = core_wait_logit
        if (
            self.control_state_dim > 0
            and self.control_wait_out is not None
            and self.control_wait_scale != 0.0
        ):
            wait_logit = wait_logit + (
                self.control_wait_scale * self.control_wait_out(updated_control.float()).to(updated_control.dtype)
            )
        release_logit = self._compute_release_logit(
            router_hidden=router_hidden,
            updated_control=updated_control,
            updated_wait=updated_wait,
        )
        if (
            release_logit is not None
            and self.release_scale != 0.0
            and self.release_gate_mode == "bias"
        ):
            wait_logit = wait_logit - (
                self.release_scale * release_logit.to(wait_logit.dtype)
            )
        return wait_logit

    def _compute_release_logit(
        self,
        *,
        router_hidden: torch.Tensor | None,
        updated_control: torch.Tensor,
        updated_wait: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.release_head is None or router_hidden is None:
            return None
        release_inputs = [router_hidden]
        if self.control_state_dim > 0:
            release_inputs.append(updated_control)
        if self.wait_state_dim > 0:
            release_inputs.append(updated_wait)
        release_features = torch.cat(release_inputs, dim=-1)
        return self.release_head(release_features.float()).to(release_features.dtype)

    def _compose_routing_logits(
        self,
        *,
        flat_logits: torch.Tensor | None,
        act_logits: torch.Tensor | None,
        wait_logit: torch.Tensor | None,
        release_logit: torch.Tensor | None,
        control_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.routing_head_mode == "flat":
            if flat_logits is None:
                raise ValueError("Flat routing requires flat logits.")
            masked_logits = flat_logits
            if self.control_state_dim > 0 and self.control_router_out is not None:
                masked_logits = masked_logits + (
                    self.control_router_scale * self.control_router_out(control_state)
                )
            return masked_logits
        if act_logits is None or wait_logit is None:
            raise ValueError(f"{self.routing_head_mode} requires factorized act/wait logits.")
        act_log_probs = F.log_softmax(act_logits.float(), dim=-1)
        if self.release_gate_mode == "direct":
            if release_logit is None:
                raise ValueError("release_gate_mode=direct requires release_logit.")
            gate_logit = self.release_gate_scale * release_logit.float()
            wait_log_prob = F.logsigmoid(-gate_logit).squeeze(-1)
            act_log_prob = F.logsigmoid(gate_logit).squeeze(-1)
        else:
            wait_log_prob = F.logsigmoid(wait_logit.float()).squeeze(-1)
            act_log_prob = F.logsigmoid(-wait_logit.float()).squeeze(-1)
        composed = torch.stack(
            [
                act_log_prob + act_log_probs[..., 0],
                act_log_prob + act_log_probs[..., 1],
                wait_log_prob,
            ],
            dim=-1,
        )
        return composed.to(act_logits.dtype)

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
