from __future__ import annotations

import math
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


def compute_task_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    task_sample_weights: torch.Tensor | None = None,
    final_query_mask: torch.Tensor | None = None,
    final_query_shaping_mode: str = "none",
    final_query_shaping_weight: float = 0.0,
    final_query_margin: float = 0.0,
    final_query_focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    per_sample_task_loss = F.cross_entropy(logits, labels, reduction="none")
    if task_sample_weights is None:
        weights = torch.ones_like(per_sample_task_loss)
    else:
        weights = task_sample_weights.to(device=logits.device, dtype=per_sample_task_loss.dtype)
    task_loss = (per_sample_task_loss * weights).sum() / weights.sum().clamp_min(eps)

    shaping_loss = torch.zeros((), device=logits.device, dtype=per_sample_task_loss.dtype)
    if (
        final_query_shaping_mode == "none"
        or final_query_shaping_weight <= 0.0
        or final_query_mask is None
    ):
        return task_loss, shaping_loss

    final_query_mask = final_query_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(final_query_mask.any().item()):
        return task_loss, shaping_loss

    if final_query_shaping_mode == "margin":
        target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = logits.masked_fill(
            F.one_hot(labels, num_classes=logits.shape[-1]).to(dtype=torch.bool),
            float("-inf"),
        )
        max_other_logits = other_logits.max(dim=-1).values
        per_sample_shaping = F.relu(float(final_query_margin) - (target_logits - max_other_logits))
    elif final_query_shaping_mode == "focal":
        target_probs = F.softmax(logits.float(), dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(eps)
        per_sample_shaping = ((1.0 - target_probs) ** float(final_query_focal_gamma)) * per_sample_task_loss.float()
    else:
        raise ValueError(f"Unknown final_query_shaping_mode: {final_query_shaping_mode}")

    shaping_weights = weights.to(per_sample_shaping.dtype) * final_query_mask.to(per_sample_shaping.dtype)
    shaping_loss = float(final_query_shaping_weight) * (
        (per_sample_shaping * shaping_weights).sum() / shaping_weights.sum().clamp_min(eps)
    )
    return task_loss + shaping_loss.to(task_loss.dtype), shaping_loss.to(task_loss.dtype)


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


class StandardizedMLPReadoutHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim > 0 else max(16, min(128, input_dim * 2))
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.input_mean.zero_()
        self.input_std.fill_(1.0)

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(1e-5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = (x - self.input_mean.to(device=x.device, dtype=x.dtype)) / self.input_std.to(
            device=x.device,
            dtype=x.dtype,
        ).clamp_min(1e-5)
        hidden = F.gelu(self.fc1(hidden))
        return self.fc2(hidden)


class CosineReadoutHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        metric_dim: int = 0,
        init_scale: float = 10.0,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.metric_dim = metric_dim if metric_dim > 0 else input_dim
        self.readout_norm = nn.LayerNorm(input_dim)
        if self.metric_dim == input_dim:
            self.readout_input_proj = None
        else:
            self.readout_input_proj = nn.Linear(input_dim, self.metric_dim)
        self.readout_prototypes = nn.Parameter(torch.empty(num_classes, self.metric_dim))
        nn.init.normal_(self.readout_prototypes, std=0.02)
        if learnable_scale:
            self.readout_logit_scale = nn.Parameter(torch.tensor(math.log(max(init_scale, 1e-3)), dtype=torch.float32))
            self._fixed_scale = None
        else:
            self.readout_logit_scale = None
            self._fixed_scale = float(init_scale)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.readout_norm(x)
        if self.readout_input_proj is not None:
            hidden = F.gelu(self.readout_input_proj(hidden))
        return F.normalize(hidden, dim=-1, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self._embed(x)
        prototypes = F.normalize(self.readout_prototypes, dim=-1, eps=1e-6)
        if self.readout_logit_scale is None:
            scale = x.new_tensor(self._fixed_scale)
        else:
            scale = self.readout_logit_scale.exp().clamp(max=100.0).to(device=x.device, dtype=x.dtype)
        return scale * embedded @ prototypes.transpose(0, 1).to(dtype=embedded.dtype)

    def prototype_pull_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self._embed(x)
        prototypes = F.normalize(self.readout_prototypes, dim=-1, eps=1e-6)
        target_proto = prototypes[labels]
        per_sample = 1.0 - (embedded * target_proto).sum(dim=-1)
        if sample_weights is None:
            weights = torch.ones_like(per_sample)
        else:
            weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


class MixtureReadoutHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        query_dim: int,
        num_heads: int = 2,
        branch_hidden_dim: int = 0,
        gate_source: str = "input",
        gate_hidden_dim: int = 0,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_heads < 2:
            raise ValueError("MixtureReadoutHead requires at least 2 heads.")
        if gate_source not in {"input", "query", "input_query"}:
            raise ValueError(f"Unknown gate_source: {gate_source}")
        self.num_heads = num_heads
        self.gate_source = gate_source
        self.temperature = max(float(temperature), 1e-3)
        self.readout_norm = nn.LayerNorm(input_dim)
        self.gate_query_proj = None
        gate_input_dim = 0
        if gate_source in {"input", "input_query"}:
            gate_input_dim += input_dim
        if gate_source in {"query", "input_query"}:
            self.gate_query_proj = nn.Sequential(
                nn.LayerNorm(query_dim),
                nn.Linear(query_dim, input_dim),
                nn.GELU(),
            )
            gate_input_dim += input_dim
        if gate_hidden_dim > 0:
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, num_heads),
            )
        else:
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, num_heads),
            )
        self.experts = nn.ModuleList()
        for _ in range(num_heads):
            if branch_hidden_dim > 0:
                self.experts.append(
                    nn.Sequential(
                        nn.Linear(input_dim, branch_hidden_dim),
                        nn.GELU(),
                        nn.Linear(branch_hidden_dim, num_classes),
                    )
                )
            else:
                self.experts.append(nn.Linear(input_dim, num_classes))
        self.last_mixture_weights: torch.Tensor | None = None

    def _gate_input(self, x: torch.Tensor, query_obs: torch.Tensor | None) -> torch.Tensor:
        parts = []
        normalized = self.readout_norm(x)
        if self.gate_source in {"input", "input_query"}:
            parts.append(normalized)
        if self.gate_source in {"query", "input_query"}:
            if query_obs is None or self.gate_query_proj is None:
                raise ValueError("query_obs is required for query-conditioned mixture gating.")
            parts.append(self.gate_query_proj(query_obs))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, *, query_obs: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.readout_norm(x)
        gate_logits = self.gate(self._gate_input(x, query_obs)) / self.temperature
        mixture_weights = torch.softmax(gate_logits, dim=-1)
        expert_logits = torch.stack([expert(normalized) for expert in self.experts], dim=1)
        self.last_mixture_weights = mixture_weights.detach()
        return (mixture_weights.unsqueeze(-1) * expert_logits).sum(dim=1)

    def balance_loss(self) -> torch.Tensor:
        if self.last_mixture_weights is None:
            raise RuntimeError("balance_loss called before forward.")
        mean_weights = self.last_mixture_weights.mean(dim=0)
        target = torch.full_like(mean_weights, 1.0 / self.num_heads)
        return ((mean_weights - target) ** 2).mean()

    def entropy(self) -> torch.Tensor:
        if self.last_mixture_weights is None:
            raise RuntimeError("entropy called before forward.")
        weights = self.last_mixture_weights.clamp_min(1e-8)
        return -(weights * weights.log()).sum(dim=-1).mean()


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
        self.readout_head_mode = str(config.get("readout_head_mode", "linear"))
        self.readout_mlp_hidden_dim = int(config.get("readout_mlp_hidden_dim", 0))
        self.readout_metric_dim = int(config.get("readout_metric_dim", 0))
        self.readout_logit_scale_init = float(config.get("readout_logit_scale_init", 10.0))
        self.readout_learnable_scale = bool(config.get("readout_learnable_scale", True))
        self.readout_prototype_pull_weight = float(config.get("readout_prototype_pull_weight", 0.0))
        self.readout_mixture_num_heads = int(config.get("readout_mixture_num_heads", 2))
        self.readout_mixture_branch_hidden_dim = int(config.get("readout_mixture_branch_hidden_dim", 0))
        self.readout_mixture_gate_source = str(config.get("readout_mixture_gate_source", "input"))
        self.readout_mixture_gate_hidden_dim = int(config.get("readout_mixture_gate_hidden_dim", 0))
        self.readout_mixture_temperature = float(config.get("readout_mixture_temperature", 1.0))
        self.readout_mixture_balance_weight = float(config.get("readout_mixture_balance_weight", 0.0))
        self.readout_views = tuple(
            str(name)
            for name in config.get(
                "readout_views",
                ["final_sink_state", "packet_state_query"],
            )
        )
        self.trajectory_bank_views = tuple(
            str(name)
            for name in config.get(
                "trajectory_bank_views",
                ["sink_state"],
            )
        )
        self.trajectory_bank_window = int(config.get("trajectory_bank_window", 8))
        self.trajectory_bank_stride = int(config.get("trajectory_bank_stride", 1))
        self.trajectory_bank_anchor = str(config.get("trajectory_bank_anchor", "final"))
        self.trajectory_bank_latent_slots = int(config.get("trajectory_bank_latent_slots", 2))
        self.trajectory_bank_route_features = tuple(
            str(name) for name in config.get("trajectory_bank_route_features", [])
        )
        self.sink_mode = str(config.get("sink_mode", "single"))
        self.sink_slots = int(config.get("sink_slots", 1))
        if self.sink_slots <= 0:
            raise ValueError("sink_slots must be positive.")
        if self.sink_mode not in {"single", "keyed_mixture"}:
            raise ValueError("sink_mode must be one of: single, keyed_mixture")
        self.trajectory_bank_use_positional_features = bool(
            config.get("trajectory_bank_use_positional_features", True)
        )
        self.factorized_content_source = str(config.get("factorized_content_source", "final_sink_state"))
        self.factorized_combiner_mode = str(config.get("factorized_combiner_mode", "concat"))
        payload_cardinality = config.get("payload_cardinality")
        self.payload_cardinality = int(self.num_classes if payload_cardinality is None else payload_cardinality)
        self.factorized_payload_aux_weight = float(config.get("factorized_payload_aux_weight", 0.0))
        self.factorized_query_aux_weight = float(config.get("factorized_query_aux_weight", 0.0))
        self.factorized_aux_final_query_only = bool(config.get("factorized_aux_final_query_only", False))
        self.query_offset = int(config.get("query_offset", -1))
        query_cardinality = config.get("query_cardinality", 0)
        self.query_cardinality = int(0 if query_cardinality is None else query_cardinality)
        self.readout_iter_steps = int(config.get("readout_iter_steps", 1))
        self.readout_view_dropout = float(config.get("readout_view_dropout", 0.0))
        self.readout_attention_heads = int(config.get("readout_attention_heads", 1))
        self.multiview_adapter_mode = str(config.get("multiview_adapter_mode", "none"))
        self.multiview_adapter_rank = int(config.get("multiview_adapter_rank", 0))
        self.multiview_adapter_hidden_dim = int(
            config.get("multiview_adapter_hidden_dim", self.hidden_dim)
        )
        self.readout_adapter_mode = str(config.get("readout_adapter_mode", "none"))
        self.readout_adapter_rank = int(config.get("readout_adapter_rank", 0))
        self.readout_adapter_hidden_dim = int(
            config.get("readout_adapter_hidden_dim", self.hidden_dim)
        )
        self.final_query_shaping_mode = str(config.get("final_query_shaping_mode", "none"))
        self.final_query_shaping_weight = float(config.get("final_query_shaping_weight", 0.0))
        self.final_query_margin = float(config.get("final_query_margin", 0.0))
        self.final_query_focal_gamma = float(config.get("final_query_focal_gamma", 2.0))
        self.multiview_readout_modes = {
            "multiview_concat",
            "multiview_query_gated",
            "multiview_query_film",
            "multiview_cross_attention",
        }
        self.temporal_bank_readout_modes = {
            "temporalbank_query_gated",
            "temporalbank_query_film",
            "temporalbank_cross_attention",
            "temporalbank_bilinear",
            "temporalbank_latent_pool",
        }
        self.factorized_readout_modes = {
            "factorized_content_query",
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
        self.sink_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.sink_slots)
        self.sink_slot_proj = None
        if self.sink_mode == "keyed_mixture":
            self.sink_slot_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.sink_slots),
            )
        self.query_readout_proj = None
        self.multiview_baseline_proj = None
        self.multiview_query_proj = None
        self.multiview_fusion = None
        self.multiview_attention = None
        self.multiview_attention_norm = None
        self.multiview_ff = None
        self.multiview_adapter = None
        self.trajectory_bank_view_proj = None
        self.trajectory_bank_query_proj = None
        self.trajectory_bank_route_proj = None
        self.trajectory_bank_attention = None
        self.trajectory_bank_attention_norm = None
        self.trajectory_bank_ff = None
        self.trajectory_bank_latents = None
        self.trajectory_bank_score = None
        self.trajectory_bank_bilinear = None
        self.trajectory_bank_film = None
        self.factorized_query_proj = None
        self.factorized_content_proj = None
        self.factorized_combiner = None
        self.factorized_gate = None
        self.factorized_bilinear = None
        self.factorized_film = None
        self.factorized_payload_head = None
        self.factorized_query_head = None
        self.multiview_view_dropout = (
            nn.Dropout(self.readout_view_dropout)
            if self.readout_view_dropout > 0.0
            else nn.Identity()
        )
        self.readout_adapter = None
        effective_base_mode = (
            self.readout_base_mode
            if (
                self.readout_mode in self.multiview_readout_modes
                or self.readout_mode in self.temporal_bank_readout_modes
                or self.readout_mode in self.factorized_readout_modes
            )
            else self.readout_mode
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
        elif self.readout_mode == "probe_query_views":
            if self.query_cardinality <= 0 or self.query_offset < 0:
                raise ValueError(
                    "probe_query_views requires positive query_cardinality and non-negative query_offset."
                )
            readout_input_dim = (self.hidden_dim * len(self.readout_views)) + self.query_cardinality
        elif self.readout_mode in self.temporal_bank_readout_modes:
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
            bank_input_dim = (self.hidden_dim * len(self.trajectory_bank_views)) + (
                2 if self.trajectory_bank_use_positional_features else 0
            )
            self.trajectory_bank_view_proj = nn.Sequential(
                nn.LayerNorm(bank_input_dim),
                nn.Linear(bank_input_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.trajectory_bank_query_proj = nn.Sequential(
                nn.LayerNorm(self.obs_dim),
                nn.Linear(self.obs_dim, self.hidden_dim),
                nn.GELU(),
            )
            route_dim = self._trajectory_route_feature_dim()
            if route_dim > 0:
                self.trajectory_bank_route_proj = nn.Sequential(
                    nn.LayerNorm(route_dim),
                    nn.Linear(route_dim, self.hidden_dim),
                    nn.GELU(),
                )
            if self.readout_mode in {"temporalbank_cross_attention", "temporalbank_latent_pool"}:
                self.trajectory_bank_attention = nn.MultiheadAttention(
                    self.hidden_dim,
                    self.readout_attention_heads,
                    batch_first=True,
                )
                self.trajectory_bank_attention_norm = nn.LayerNorm(self.hidden_dim)
                self.trajectory_bank_ff = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                )
                if self.readout_mode == "temporalbank_latent_pool":
                    self.trajectory_bank_latents = nn.Parameter(
                        torch.randn(self.trajectory_bank_latent_slots, self.hidden_dim) * 0.02
                    )
            else:
                if self.readout_mode == "temporalbank_bilinear":
                    self.trajectory_bank_bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, 1)
                else:
                    self.trajectory_bank_score = nn.Linear(self.hidden_dim, 1)
                if self.readout_mode == "temporalbank_query_film":
                    self.trajectory_bank_film = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            readout_input_dim = self.hidden_dim
        elif self.readout_mode in self.factorized_readout_modes:
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
            if self.factorized_content_source == "trajectory_bank":
                bank_input_dim = (self.hidden_dim * len(self.trajectory_bank_views)) + (
                    2 if self.trajectory_bank_use_positional_features else 0
                )
                self.trajectory_bank_view_proj = nn.Sequential(
                    nn.LayerNorm(bank_input_dim),
                    nn.Linear(bank_input_dim, self.hidden_dim),
                    nn.GELU(),
                )
                self.trajectory_bank_query_proj = nn.Sequential(
                    nn.LayerNorm(self.obs_dim),
                    nn.Linear(self.obs_dim, self.hidden_dim),
                    nn.GELU(),
                )
                route_dim = self._trajectory_route_feature_dim()
                if route_dim > 0:
                    self.trajectory_bank_route_proj = nn.Sequential(
                        nn.LayerNorm(route_dim),
                        nn.Linear(route_dim, self.hidden_dim),
                        nn.GELU(),
                    )
                self.trajectory_bank_score = nn.Linear(self.hidden_dim, 1)
            self.factorized_query_proj = nn.Sequential(
                nn.LayerNorm(self.obs_dim),
                nn.Linear(self.obs_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.factorized_content_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            if self.factorized_combiner_mode == "concat":
                self.factorized_combiner = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                )
            elif self.factorized_combiner_mode == "gated":
                self.factorized_gate = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                self.factorized_combiner = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.GELU(),
                )
            elif self.factorized_combiner_mode == "bilinear":
                self.factorized_bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, self.hidden_dim)
            elif self.factorized_combiner_mode == "film":
                self.factorized_film = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            else:
                raise ValueError(f"Unknown factorized_combiner_mode: {self.factorized_combiner_mode}")
            if self.factorized_payload_aux_weight > 0.0:
                self.factorized_payload_head = nn.Linear(self.hidden_dim, self.payload_cardinality)
            if self.factorized_query_aux_weight > 0.0:
                self.factorized_query_head = nn.Linear(
                    self.hidden_dim,
                    max(1, self.query_cardinality if self.query_cardinality > 0 else self.num_classes),
                )
            readout_input_dim = self.hidden_dim
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

        if self.multiview_adapter_mode == "none":
            self.multiview_adapter = None
        elif self.multiview_adapter_mode == "low_rank":
            rank = max(1, self.multiview_adapter_rank)
            self.multiview_adapter = LowRankAdapter(self.hidden_dim, rank)
        elif self.multiview_adapter_mode == "residual_mlp":
            hidden = max(1, self.multiview_adapter_hidden_dim)
            self.multiview_adapter = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.hidden_dim),
            )
            nn.init.zeros_(self.multiview_adapter[-1].weight)
            nn.init.zeros_(self.multiview_adapter[-1].bias)
        elif self.multiview_adapter_mode == "affine":
            self.multiview_adapter = AffineAdapter(self.hidden_dim)
        else:
            raise ValueError(f"Unknown multiview_adapter_mode: {self.multiview_adapter_mode}")

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
        if self.readout_head_mode == "linear":
            self.readout = nn.Sequential(
                nn.LayerNorm(readout_input_dim),
                nn.Linear(readout_input_dim, self.num_classes),
            )
        elif self.readout_head_mode == "mlp":
            self.readout = StandardizedMLPReadoutHead(
                readout_input_dim,
                self.num_classes,
                hidden_dim=self.readout_mlp_hidden_dim,
            )
        elif self.readout_head_mode == "cosine":
            self.readout = CosineReadoutHead(
                readout_input_dim,
                self.num_classes,
                metric_dim=self.readout_metric_dim,
                init_scale=self.readout_logit_scale_init,
                learnable_scale=self.readout_learnable_scale,
            )
        elif self.readout_head_mode == "mixture":
            self.readout = MixtureReadoutHead(
                readout_input_dim,
                self.num_classes,
                query_dim=self.obs_dim,
                num_heads=self.readout_mixture_num_heads,
                branch_hidden_dim=self.readout_mixture_branch_hidden_dim,
                gate_source=self.readout_mixture_gate_source,
                gate_hidden_dim=self.readout_mixture_gate_hidden_dim,
                temperature=self.readout_mixture_temperature,
            )
        else:
            raise ValueError(f"Unknown readout_head_mode: {self.readout_head_mode}")

    def _trajectory_route_feature_dim(self) -> int:
        dim = 0
        for name in self.trajectory_bank_route_features:
            if name in {"action_histogram", "route_action_histogram"}:
                dim += len(ACTION_NAMES)
            else:
                dim += 1
        return dim

    def _sink_features(self, packet_state: torch.Tensor) -> torch.Tensor:
        projected = self.sink_proj(packet_state)
        if self.sink_mode == "single":
            return projected
        if self.sink_slot_proj is None:
            raise RuntimeError("sink_slot_proj must be configured for keyed_mixture sink mode.")
        slot_logits = self.sink_slot_proj(packet_state)
        slot_weights = torch.softmax(slot_logits, dim=-1)
        projected = projected.view(*packet_state.shape[:-1], self.sink_slots, self.hidden_dim)
        return (slot_weights.unsqueeze(-1) * projected).sum(dim=-2)

    def _baseline_readout_input_from_query(
        self,
        sink_state: torch.Tensor,
        query_obs: torch.Tensor,
        *,
        for_multiview: bool = False,
    ) -> torch.Tensor:
        mode = (
            self.readout_mode
            if (
                self.readout_mode not in self.multiview_readout_modes
                and self.readout_mode not in self.temporal_bank_readout_modes
                and self.readout_mode not in self.factorized_readout_modes
                and self.readout_mode != "probe_query_views"
            )
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

    def _baseline_readout_input(
        self,
        sink_state: torch.Tensor,
        observations: torch.Tensor,
        *,
        for_multiview: bool = False,
    ) -> torch.Tensor:
        query_obs = observations[:, -1, 0]
        return self._baseline_readout_input_from_query(
            sink_state,
            query_obs,
            for_multiview=for_multiview,
        )

    def _query_one_hot_view(self, observations: torch.Tensor) -> torch.Tensor:
        end = self.query_offset + self.query_cardinality
        if self.query_offset < 0 or self.query_cardinality <= 0 or end > self.obs_dim:
            raise ValueError(
                "Query view requested, but query_offset/query_cardinality are not configured for the benchmark."
            )
        return observations[:, -1, 0, self.query_offset:end]

    def _apply_multiview_adapter(self, fused: torch.Tensor) -> torch.Tensor:
        if self.multiview_adapter is None or fused.shape[-1] != self.hidden_dim:
            return fused
        return fused + self.multiview_adapter(fused)

    def _apply_readout_adapter(self, fused: torch.Tensor) -> torch.Tensor:
        if self.readout_adapter is None or fused.shape[-1] != self.hidden_dim:
            return fused
        return fused + self.readout_adapter(fused)

    def _trajectory_anchor_index(
        self,
        *,
        trace: dict[str, torch.Tensor],
        first_exit_time: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        anchor = self.trajectory_bank_anchor
        if anchor in {"final", "final_query"}:
            index = torch.full_like(first_exit_time, seq_len - 1)
        elif anchor == "exit":
            index = first_exit_time.round()
        elif anchor == "delay_peak":
            index = trace["action_mass"][:, :, ACTION_DELAY].argmax(dim=1).to(first_exit_time.dtype)
        else:
            raise ValueError(f"Unknown trajectory_bank_anchor: {anchor}")
        return index.clamp_(0.0, float(seq_len - 1)).long()

    def _trajectory_route_features(
        self,
        *,
        first_exit_time: torch.Tensor,
        delays: torch.Tensor,
        action_totals: torch.Tensor,
        route_entropy_sum: torch.Tensor,
        route_conf_sum: torch.Tensor,
        route_weight_total: torch.Tensor,
        seq_len: int,
        eps: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.trajectory_bank_route_features:
            return None
        features: list[torch.Tensor] = []
        for name in self.trajectory_bank_route_features:
            if name == "exit_time":
                features.append((first_exit_time / max(1, seq_len - 1)).unsqueeze(-1))
            elif name in {"delay_fraction", "delay_counts"}:
                features.append((delays / max(1, seq_len)).unsqueeze(-1))
            elif name in {"action_histogram", "route_action_histogram"}:
                histogram = action_totals / action_totals.sum(dim=-1, keepdim=True).clamp_min(float(eps))
                features.append(histogram)
            elif name == "route_entropy":
                features.append((route_entropy_sum / route_weight_total.clamp_min(float(eps))).unsqueeze(-1))
            elif name == "route_confidence":
                features.append((route_conf_sum / route_weight_total.clamp_min(float(eps))).unsqueeze(-1))
            else:
                raise ValueError(f"Unknown trajectory bank route feature: {name}")
        return torch.cat(features, dim=-1)

    def _select_temporal_bank(
        self,
        *,
        trace: dict[str, torch.Tensor],
        observations: torch.Tensor,
        first_exit_time: torch.Tensor,
        delays: torch.Tensor,
        action_totals: torch.Tensor,
        route_entropy_sum: torch.Tensor,
        route_conf_sum: torch.Tensor,
        route_weight_total: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = observations.shape[:2]
        device = observations.device
        dtype = observations.dtype
        anchor_idx = self._trajectory_anchor_index(
            trace=trace,
            first_exit_time=first_exit_time,
            seq_len=seq_len,
        )
        query_obs = observations[:, -1, 0]

        view_tensors: list[torch.Tensor] = []
        for name in self.trajectory_bank_views:
            if name == "sink_state":
                view_tensors.append(trace["sink_state"])
            elif name == "packet_state":
                view_tensors.append(trace["packet_state"])
            elif name == "temporal_baseline_readout":
                query_seq = query_obs.unsqueeze(1).expand(-1, seq_len, -1)
                view_tensors.append(
                    self._baseline_readout_input_from_query(
                        trace["sink_state"],
                        query_seq,
                        for_multiview=False,
                    )
                )
            else:
                raise ValueError(f"Unknown trajectory bank view: {name}")
        bank_input = torch.cat(view_tensors, dim=-1)
        if self.trajectory_bank_use_positional_features:
            position = torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=dtype).view(1, seq_len, 1)
            relative = (
                torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len)
                - anchor_idx.to(dtype).unsqueeze(1)
            ) / max(1, seq_len - 1)
            bank_input = torch.cat(
                [
                    bank_input,
                    position.expand(batch_size, -1, -1),
                    relative.unsqueeze(-1),
                ],
                dim=-1,
            )
        bank = self.trajectory_bank_view_proj(bank_input)
        window = max(1, min(self.trajectory_bank_window, seq_len))
        stride = max(1, self.trajectory_bank_stride)
        offsets = torch.arange(window, device=device)
        offsets = (window - 1 - offsets) * stride
        gather_index = (anchor_idx.unsqueeze(1) - offsets.unsqueeze(0)).clamp_(0, seq_len - 1)
        bank = bank.gather(1, gather_index.unsqueeze(-1).expand(-1, -1, bank.shape[-1]))
        route_features = self._trajectory_route_features(
            first_exit_time=first_exit_time,
            delays=delays,
            action_totals=action_totals,
            route_entropy_sum=route_entropy_sum,
            route_conf_sum=route_conf_sum,
            route_weight_total=route_weight_total,
            seq_len=seq_len,
            eps=eps,
        )
        return bank, route_features

    def _temporal_bank_context(
        self,
        *,
        bank: torch.Tensor,
        query_obs: torch.Tensor,
        route_features: torch.Tensor | None,
    ) -> torch.Tensor:
        query_token = self.trajectory_bank_query_proj(query_obs)
        if route_features is not None:
            query_token = query_token + self.trajectory_bank_route_proj(route_features)
        if self.readout_mode == "temporalbank_cross_attention":
            latent = query_token
            for _ in range(max(1, self.readout_iter_steps)):
                attended, _ = self.trajectory_bank_attention(
                    latent.unsqueeze(1),
                    bank,
                    bank,
                    need_weights=False,
                )
                latent = self.trajectory_bank_attention_norm(latent + attended.squeeze(1))
                latent = latent + self.trajectory_bank_ff(latent)
            return latent
        if self.readout_mode == "temporalbank_latent_pool":
            latent_bank = self.trajectory_bank_latents.unsqueeze(0).expand(bank.shape[0], -1, -1)
            latent_bank = latent_bank + query_token.unsqueeze(1)
            for _ in range(max(1, self.readout_iter_steps)):
                attended, _ = self.trajectory_bank_attention(
                    latent_bank,
                    bank,
                    bank,
                    need_weights=False,
                )
                latent_bank = self.trajectory_bank_attention_norm(latent_bank + attended)
                latent_bank = latent_bank + self.trajectory_bank_ff(latent_bank)
            latent_scores = torch.einsum("bld,bd->bl", latent_bank, query_token)
            latent_weights = torch.softmax(latent_scores, dim=-1)
            return (latent_weights.unsqueeze(-1) * latent_bank).sum(dim=1)
        if self.readout_mode == "temporalbank_query_film":
            film_params = self.trajectory_bank_film(query_token)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            bank = bank * (1.0 + torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        if self.readout_mode == "temporalbank_bilinear":
            query_bank = query_token.unsqueeze(1).expand(-1, bank.shape[1], -1)
            scores = self.trajectory_bank_bilinear(bank, query_bank).squeeze(-1)
        else:
            scores = self.trajectory_bank_score(torch.tanh(bank + query_token.unsqueeze(1))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * bank).sum(dim=1)

    def _factorized_readout_input(
        self,
        *,
        sink_state: torch.Tensor,
        observations: torch.Tensor,
        trace: dict[str, torch.Tensor],
        first_exit_time: torch.Tensor,
        delays: torch.Tensor,
        action_totals: torch.Tensor,
        route_entropy_sum: torch.Tensor,
        route_conf_sum: torch.Tensor,
        route_weight_total: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_obs = observations[:, -1, 0]
        if self.factorized_content_source == "trajectory_bank":
            bank, route_features = self._select_temporal_bank(
                trace=trace,
                observations=observations,
                first_exit_time=first_exit_time,
                delays=delays,
                action_totals=action_totals,
                route_entropy_sum=route_entropy_sum,
                route_conf_sum=route_conf_sum,
                route_weight_total=route_weight_total,
                eps=eps,
            )
            content = self._temporal_bank_context(
                bank=bank,
                query_obs=query_obs,
                route_features=route_features,
            )
        elif self.factorized_content_source == "final_sink_state":
            content = sink_state
        else:
            raise ValueError(f"Unknown factorized_content_source: {self.factorized_content_source}")
        content = self.factorized_content_proj(content)
        query_hidden = self.factorized_query_proj(query_obs)
        if self.factorized_combiner_mode == "concat":
            return self.factorized_combiner(torch.cat([content, query_hidden], dim=-1)), content, query_hidden
        if self.factorized_combiner_mode == "gated":
            gated = content * torch.sigmoid(self.factorized_gate(query_hidden))
            return self.factorized_combiner(torch.cat([gated, query_hidden], dim=-1)), content, query_hidden
        if self.factorized_combiner_mode == "bilinear":
            return torch.tanh(self.factorized_bilinear(content, query_hidden)), content, query_hidden
        if self.factorized_combiner_mode == "film":
            gamma, beta = torch.chunk(self.factorized_film(query_hidden), 2, dim=-1)
            return content * (1.0 + torch.tanh(gamma)) + beta, content, query_hidden
        raise ValueError(f"Unknown factorized_combiner_mode: {self.factorized_combiner_mode}")

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
        factorized_payload_targets: torch.Tensor | None = None,
        factorized_query_targets: torch.Tensor | None = None,
        task_sample_weights: torch.Tensor | None = None,
        final_query_mask: torch.Tensor | None = None,
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

        collect_internal_trace = (
            return_trace
            or self.readout_mode in self.temporal_bank_readout_modes
            or self.readout_mode in self.factorized_readout_modes
        )
        trace: dict[str, torch.Tensor] | None = None
        if collect_internal_trace:
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

                exit_features = self._sink_features(packet_next)
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
            sink_state = sink_state + (packet_masses.unsqueeze(-1) * self._sink_features(packet_states)).sum(dim=1)
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

        factorized_content_hidden = None
        factorized_query_hidden = None
        if self.readout_mode == "probe_query_views":
            baseline_view = self._baseline_readout_input(
                sink_state,
                observations,
                for_multiview=False,
            )
            view_map = {
                "final_sink_state": sink_state,
                "sink_state_query": sink_state_query,
                "packet_state_query": packet_state_query,
                "baseline_readout_input": baseline_view,
            }
            view_tensors = [view_map[name] for name in self.readout_views]
            readout_input = torch.cat([*view_tensors, self._query_one_hot_view(observations)], dim=-1)
        elif self.readout_mode in self.multiview_readout_modes:
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
                readout_input = self._apply_multiview_adapter(latent)
            else:
                fused = self.multiview_fusion(
                    torch.cat(
                        [self.multiview_view_dropout(view) for view in view_tensors],
                        dim=-1,
                    )
                )
                fused = self._apply_multiview_adapter(fused)
                if self.readout_mode == "multiview_query_gated":
                    query_gate = torch.sigmoid(self.multiview_query_proj(observations[:, -1, 0]))
                    fused = fused * query_gate
                elif self.readout_mode == "multiview_query_film":
                    film_params = self.multiview_query_proj(observations[:, -1, 0])
                    gamma, beta = torch.chunk(film_params, 2, dim=-1)
                    fused = fused * (1.0 + torch.tanh(gamma)) + beta
                readout_input = fused
            readout_input = self._apply_readout_adapter(readout_input)
        elif self.readout_mode in self.temporal_bank_readout_modes:
            if trace is None:
                raise RuntimeError("Temporal-bank readout requires trace collection.")
            baseline_view = self._baseline_readout_input(
                sink_state,
                observations,
                for_multiview=False,
            )
            bank, route_features = self._select_temporal_bank(
                trace=trace,
                observations=observations,
                first_exit_time=first_exit_time,
                delays=delays,
                action_totals=action_totals,
                route_entropy_sum=route_entropy_sum,
                route_conf_sum=route_conf_sum,
                route_weight_total=route_weight_total,
                eps=eps,
            )
            readout_input = self._temporal_bank_context(
                bank=bank,
                query_obs=observations[:, -1, 0],
                route_features=route_features,
            )
            readout_input = self._apply_readout_adapter(readout_input)
        elif self.readout_mode in self.factorized_readout_modes:
            if trace is None:
                raise RuntimeError("Factorized readout requires trace collection.")
            baseline_view = self._baseline_readout_input(
                sink_state,
                observations,
                for_multiview=False,
            )
            readout_input, factorized_content_hidden, factorized_query_hidden = self._factorized_readout_input(
                sink_state=sink_state,
                observations=observations,
                trace=trace,
                first_exit_time=first_exit_time,
                delays=delays,
                action_totals=action_totals,
                route_entropy_sum=route_entropy_sum,
                route_conf_sum=route_conf_sum,
                route_weight_total=route_weight_total,
                eps=eps,
            )
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
            if factorized_content_hidden is not None:
                trace["factorized_content_hidden"] = factorized_content_hidden
            if factorized_query_hidden is not None:
                trace["factorized_query_hidden"] = factorized_query_hidden
            trace["final_readout_input"] = readout_input
        if self.readout_head_mode == "mixture":
            logits = self.readout(readout_input, query_obs=observations[:, -1, 0])
        else:
            logits = self.readout(readout_input)
        task_loss, task_shaping_loss = compute_task_classification_loss(
            logits,
            labels,
            task_sample_weights=task_sample_weights,
            final_query_mask=final_query_mask,
            final_query_shaping_mode=self.final_query_shaping_mode,
            final_query_shaping_weight=self.final_query_shaping_weight,
            final_query_margin=self.final_query_margin,
            final_query_focal_gamma=self.final_query_focal_gamma,
        )
        prototype_pull_loss = torch.zeros((), device=device, dtype=dtype)
        if self.readout_prototype_pull_weight > 0.0 and hasattr(self.readout, "prototype_pull_loss"):
            prototype_pull_loss = self.readout.prototype_pull_loss(
                readout_input,
                labels,
                sample_weights=task_sample_weights,
            ).to(dtype)
            task_loss = task_loss + (self.readout_prototype_pull_weight * prototype_pull_loss)
        mixture_balance_loss = torch.zeros((), device=device, dtype=dtype)
        mixture_gate_entropy = torch.zeros((), device=device, dtype=dtype)
        mixture_top1_weight = torch.zeros((), device=device, dtype=dtype)
        if self.readout_mixture_balance_weight > 0.0 and hasattr(self.readout, "balance_loss"):
            mixture_balance_loss = self.readout.balance_loss().to(device=device, dtype=dtype)
            task_loss = task_loss + (self.readout_mixture_balance_weight * mixture_balance_loss)
        if hasattr(self.readout, "entropy"):
            mixture_gate_entropy = self.readout.entropy().to(device=device, dtype=dtype)
        if getattr(self.readout, "last_mixture_weights", None) is not None:
            mixture_top1_weight = (
                self.readout.last_mixture_weights.max(dim=-1).values.mean().to(device=device, dtype=dtype)
            )
        factorized_payload_aux_loss = torch.zeros((), device=device, dtype=dtype)
        if (
            self.factorized_payload_head is not None
            and factorized_content_hidden is not None
            and factorized_payload_targets is not None
            and self.factorized_payload_aux_weight > 0.0
        ):
            aux_weights = torch.ones_like(labels, device=device, dtype=dtype)
            if self.factorized_aux_final_query_only and final_query_mask is not None:
                aux_weights = final_query_mask.to(device=device, dtype=dtype)
            if bool(aux_weights.sum().item() > 0):
                payload_logits = self.factorized_payload_head(factorized_content_hidden.float())
                payload_targets = factorized_payload_targets.to(device=device, dtype=torch.long)
                per_sample_payload_loss = F.cross_entropy(payload_logits, payload_targets, reduction="none").to(dtype)
                factorized_payload_aux_loss = (per_sample_payload_loss * aux_weights).sum() / aux_weights.sum().clamp_min(float(eps))
                task_loss = task_loss + (self.factorized_payload_aux_weight * factorized_payload_aux_loss)
        factorized_query_aux_loss = torch.zeros((), device=device, dtype=dtype)
        if (
            self.factorized_query_head is not None
            and factorized_query_hidden is not None
            and factorized_query_targets is not None
            and self.factorized_query_aux_weight > 0.0
        ):
            aux_weights = torch.ones_like(labels, device=device, dtype=dtype)
            if self.factorized_aux_final_query_only and final_query_mask is not None:
                aux_weights = final_query_mask.to(device=device, dtype=dtype)
            if bool(aux_weights.sum().item() > 0):
                query_logits = self.factorized_query_head(factorized_query_hidden.float())
                query_targets = factorized_query_targets.to(device=device, dtype=torch.long)
                per_sample_query_loss = F.cross_entropy(query_logits, query_targets, reduction="none").to(dtype)
                factorized_query_aux_loss = (per_sample_query_loss * aux_weights).sum() / aux_weights.sum().clamp_min(float(eps))
                task_loss = task_loss + (self.factorized_query_aux_weight * factorized_query_aux_loss)
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
            "task_shaping_loss": torch.full_like(accuracy, float(task_shaping_loss.detach().item())),
            "prototype_pull_loss": torch.full_like(accuracy, float(prototype_pull_loss.detach().item())),
            "factorized_payload_aux_loss": torch.full_like(accuracy, float(factorized_payload_aux_loss.detach().item())),
            "factorized_query_aux_loss": torch.full_like(accuracy, float(factorized_query_aux_loss.detach().item())),
            "mixture_balance_loss": torch.full_like(accuracy, float(mixture_balance_loss.detach().item())),
            "mixture_gate_entropy": torch.full_like(accuracy, float(mixture_gate_entropy.detach().item())),
            "mixture_top1_weight": torch.full_like(accuracy, float(mixture_top1_weight.detach().item())),
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
            trace=trace if return_trace else None,
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
