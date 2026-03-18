from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from src.models import ACTION_DELAY, ACTION_EXIT


MODE_EASY_EXIT = 0
MODE_SPATIAL_K = 1
MODE_DELAY_1 = 2
MODE_DELAY_1_THEN_SPATIAL_K = 3

MODE_NAMES = {
    MODE_EASY_EXIT: "easy_exit",
    MODE_SPATIAL_K: "spatial_k",
    MODE_DELAY_1: "delay_1",
    MODE_DELAY_1_THEN_SPATIAL_K: "delay_1_then_spatial_k",
}

LONG_MEMORY_MODE_EASY_EXIT = 0
LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT = 1
LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY = 2

LONG_MEMORY_MODE_NAMES = {
    LONG_MEMORY_MODE_EASY_EXIT: "easy_exit",
    LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT: "delay_to_trigger_exit",
    LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY: "delay_to_final_query",
}


@dataclass
class BenchmarkBatch:
    observations: torch.Tensor
    labels: torch.Tensor
    modes: torch.Tensor
    oracle_hops: torch.Tensor
    oracle_delays: torch.Tensor
    oracle_exit_time: torch.Tensor
    oracle_depth: torch.Tensor
    oracle_actions: torch.Tensor | None = None
    oracle_action_mask: torch.Tensor | None = None
    delay_write_targets: torch.Tensor | None = None
    delay_write_mask: torch.Tensor | None = None
    metadata: dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "BenchmarkBatch":
        return BenchmarkBatch(
            observations=self.observations.to(device),
            labels=self.labels.to(device),
            modes=self.modes.to(device),
            oracle_hops=self.oracle_hops.to(device),
            oracle_delays=self.oracle_delays.to(device),
            oracle_exit_time=self.oracle_exit_time.to(device),
            oracle_depth=self.oracle_depth.to(device),
            oracle_actions=self.oracle_actions.to(device) if self.oracle_actions is not None else None,
            oracle_action_mask=self.oracle_action_mask.to(device) if self.oracle_action_mask is not None else None,
            delay_write_targets=self.delay_write_targets.to(device) if self.delay_write_targets is not None else None,
            delay_write_mask=self.delay_write_mask.to(device) if self.delay_write_mask is not None else None,
            metadata={key: value.to(device) for key, value in self.metadata.items()},
        )


class SyntheticBenchmark:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.num_nodes = int(config["num_nodes"])
        self.obs_dim = int(config["obs_dim"])
        self.num_classes = int(config["num_classes"])
        self.train_seed = int(config.get("train_seed", 11))
        self.val_seed = int(config.get("val_seed", 101))
        self.test_seed = int(config.get("test_seed", 1001))
        self.noise_std = float(config.get("noise_std", 0.05))
        self.mode_names: dict[int, str] = {}

    def sample_batch(
        self,
        batch_size: int,
        split: str,
        step: int,
        device: torch.device | str = "cpu",
    ) -> BenchmarkBatch:
        generator = torch.Generator(device="cpu")
        base_seed = {
            "train": self.train_seed,
            "val": self.val_seed,
            "test": self.test_seed,
        }[split]
        generator.manual_seed(base_seed + step)
        batch = self._sample(batch_size=batch_size, generator=generator)
        return batch.to(device)

    def _sample(self, batch_size: int, generator: torch.Generator) -> BenchmarkBatch:
        raise NotImplementedError


class MixedOracleRoutingBenchmark(SyntheticBenchmark):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seq_len = int(config.get("seq_len", 2))
        if self.seq_len < 2:
            raise ValueError("MixedOracleRoutingBenchmark requires seq_len >= 2.")
        self.signal_scale = float(config.get("signal_scale", 1.5))
        self.mode_hint_dim = 4
        self.depth_hint_offset = 4
        self.depth_hint_dim = self.num_nodes
        self.label_offset = self.depth_hint_offset + self.depth_hint_dim
        self.mode_names = MODE_NAMES
        min_dim = self.label_offset + self.num_classes
        if self.obs_dim < min_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} is too small; need at least {min_dim}."
            )

    def _sample(self, batch_size: int, generator: torch.Generator) -> BenchmarkBatch:
        obs = torch.randn(
            batch_size,
            self.seq_len,
            self.num_nodes,
            self.obs_dim,
            generator=generator,
        ) * self.noise_std
        modes = torch.randint(0, 4, (batch_size,), generator=generator)
        labels = torch.randint(0, self.num_classes, (batch_size,), generator=generator)
        depths = torch.randint(
            low=1,
            high=self.num_nodes,
            size=(batch_size,),
            generator=generator,
        )

        oracle_hops = torch.zeros(batch_size, dtype=torch.long)
        oracle_delays = torch.zeros(batch_size, dtype=torch.long)
        oracle_exit_time = torch.zeros(batch_size, dtype=torch.long)
        oracle_depth = torch.zeros(batch_size, dtype=torch.long)

        mode_one_hot = F.one_hot(modes, num_classes=4).float()
        depth_one_hot = F.one_hot(depths, num_classes=self.num_nodes).float()
        label_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        obs[:, 0, 0, : self.mode_hint_dim] = mode_one_hot
        obs[:, 0, 0, self.depth_hint_offset : self.depth_hint_offset + self.depth_hint_dim] = depth_one_hot

        easy_mask = modes == MODE_EASY_EXIT
        if easy_mask.any():
            obs[easy_mask, 0, 0, self.label_offset : self.label_offset + self.num_classes] = (
                self.signal_scale * label_one_hot[easy_mask]
            )

        spatial_mask = modes == MODE_SPATIAL_K
        if spatial_mask.any():
            idx = torch.nonzero(spatial_mask, as_tuple=False).squeeze(-1)
            obs[idx, 0, depths[idx], self.label_offset : self.label_offset + self.num_classes] = (
                self.signal_scale * label_one_hot[idx]
            )
            oracle_hops[idx] = depths[idx]
            oracle_depth[idx] = depths[idx]

        delay_mask = modes == MODE_DELAY_1
        if delay_mask.any():
            idx = torch.nonzero(delay_mask, as_tuple=False).squeeze(-1)
            obs[idx, 1, 0, self.label_offset : self.label_offset + self.num_classes] = (
                self.signal_scale * label_one_hot[idx]
            )
            oracle_delays[idx] = 1
            oracle_exit_time[idx] = 1

        delay_spatial_mask = modes == MODE_DELAY_1_THEN_SPATIAL_K
        if delay_spatial_mask.any():
            idx = torch.nonzero(delay_spatial_mask, as_tuple=False).squeeze(-1)
            obs[idx, 1, depths[idx], self.label_offset : self.label_offset + self.num_classes] = (
                self.signal_scale * label_one_hot[idx]
            )
            oracle_hops[idx] = depths[idx]
            oracle_delays[idx] = 1
            oracle_exit_time[idx] = 1
            oracle_depth[idx] = depths[idx]

        return BenchmarkBatch(
            observations=obs,
            labels=labels,
            modes=modes,
            oracle_hops=oracle_hops,
            oracle_delays=oracle_delays,
            oracle_exit_time=oracle_exit_time,
            oracle_depth=oracle_depth,
            metadata={"depth": depths},
        )


class LongHorizonMemoryBenchmarkV1(SyntheticBenchmark):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seq_len = int(config["seq_len"])
        self.payload_cardinality = int(config.get("payload_cardinality", self.num_classes))
        self.query_cardinality = int(config.get("query_cardinality", self.num_classes))
        self.trigger_offset = 0
        self.query_flag_offset = 1
        self.payload_offset = 2
        self.query_offset = self.payload_offset + self.payload_cardinality
        min_dim = self.query_offset + self.query_cardinality
        if self.obs_dim < min_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} is too small; need at least {min_dim}."
            )
        if self.num_classes > max(self.payload_cardinality, self.query_cardinality):
            raise ValueError("num_classes must be <= payload/query cardinality.")

    def _sample(self, batch_size: int, generator: torch.Generator) -> BenchmarkBatch:
        obs = torch.randn(
            batch_size,
            self.seq_len,
            self.num_nodes,
            self.obs_dim,
            generator=generator,
        ) * self.noise_std

        trigger_times = torch.randint(
            low=0,
            high=max(1, self.seq_len // 2),
            size=(batch_size,),
            generator=generator,
        )
        payloads = torch.randint(
            low=0,
            high=self.payload_cardinality,
            size=(batch_size,),
            generator=generator,
        )
        queries = torch.randint(
            low=0,
            high=self.query_cardinality,
            size=(batch_size,),
            generator=generator,
        )
        labels = torch.remainder(payloads + queries, self.num_classes)

        payload_one_hot = F.one_hot(payloads, num_classes=self.payload_cardinality).float()
        query_one_hot = F.one_hot(queries, num_classes=self.query_cardinality).float()

        batch_index = torch.arange(batch_size)
        obs[batch_index, trigger_times, 0, self.trigger_offset] = 1.0
        obs[batch_index, trigger_times, 0, self.payload_offset : self.payload_offset + self.payload_cardinality] = (
            1.5 * payload_one_hot
        )
        obs[:, self.seq_len - 1, 0, self.query_flag_offset] = 1.0
        obs[:, self.seq_len - 1, 0, self.query_offset : self.query_offset + self.query_cardinality] = (
            1.5 * query_one_hot
        )

        oracle_actions = torch.full((batch_size, self.seq_len), ACTION_DELAY, dtype=torch.long)
        oracle_actions[:, self.seq_len - 1] = ACTION_EXIT
        oracle_action_mask = torch.ones(batch_size, self.seq_len, dtype=torch.float32)
        delay_write_targets = torch.zeros(batch_size, self.seq_len, dtype=torch.float32)
        delay_write_mask = torch.ones(batch_size, self.seq_len, dtype=torch.float32)
        delay_write_targets[batch_index, trigger_times] = 1.0
        oracle_delays = torch.full((batch_size,), self.seq_len - 1, dtype=torch.long)
        oracle_exit_time = torch.full((batch_size,), self.seq_len - 1, dtype=torch.long)
        retrieval_distance = (self.seq_len - 1 - trigger_times).long()

        return BenchmarkBatch(
            observations=obs,
            labels=labels.long(),
            modes=torch.full((batch_size,), LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY, dtype=torch.long),
            oracle_hops=torch.zeros(batch_size, dtype=torch.long),
            oracle_delays=oracle_delays,
            oracle_exit_time=oracle_exit_time,
            oracle_depth=torch.zeros(batch_size, dtype=torch.long),
            oracle_actions=oracle_actions,
            oracle_action_mask=oracle_action_mask,
            delay_write_targets=delay_write_targets,
            delay_write_mask=delay_write_mask,
            metadata={
                "trigger_time": trigger_times.long(),
                "query_time": torch.full_like(trigger_times, self.seq_len - 1),
                "retrieval_distance": retrieval_distance,
                "payload": payloads.long(),
                "query": queries.long(),
            },
        )


class LongHorizonMemoryBenchmarkV2(SyntheticBenchmark):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seq_len = int(config["seq_len"])
        self.payload_cardinality = int(config.get("payload_cardinality", self.num_classes))
        self.query_cardinality = int(config.get("query_cardinality", self.num_classes))
        self.mode_probs = torch.tensor(
            config.get("mode_probs", [0.25, 0.25, 0.5]),
            dtype=torch.float32,
        )
        if self.mode_probs.numel() != 3:
            raise ValueError("mode_probs must have length 3 for Benchmark B v2.")
        self.mode_probs = self.mode_probs / self.mode_probs.sum().clamp_min(1e-6)

        self.trigger_offset = 0
        self.query_flag_offset = 1
        self.mode_offset = 2
        self.mode_dim = 3
        self.payload_offset = self.mode_offset + self.mode_dim
        self.query_offset = self.payload_offset + self.payload_cardinality
        min_dim = self.query_offset + self.query_cardinality
        self.mode_names = LONG_MEMORY_MODE_NAMES
        if self.obs_dim < min_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} is too small; need at least {min_dim}."
            )

    def _sample(self, batch_size: int, generator: torch.Generator) -> BenchmarkBatch:
        obs = torch.randn(
            batch_size,
            self.seq_len,
            self.num_nodes,
            self.obs_dim,
            generator=generator,
        ) * self.noise_std

        modes = torch.multinomial(self.mode_probs, num_samples=batch_size, replacement=True, generator=generator)
        trigger_times = torch.randint(
            low=1,
            high=max(2, self.seq_len // 2),
            size=(batch_size,),
            generator=generator,
        )
        payloads = torch.randint(
            low=0,
            high=self.payload_cardinality,
            size=(batch_size,),
            generator=generator,
        )
        queries = torch.randint(
            low=0,
            high=self.query_cardinality,
            size=(batch_size,),
            generator=generator,
        )
        labels = torch.zeros(batch_size, dtype=torch.long)

        oracle_actions = torch.full((batch_size, self.seq_len), ACTION_DELAY, dtype=torch.long)
        oracle_action_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.float32)
        delay_write_targets = torch.zeros(batch_size, self.seq_len, dtype=torch.float32)
        delay_write_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.float32)
        oracle_delays = torch.zeros(batch_size, dtype=torch.long)
        oracle_exit_time = torch.zeros(batch_size, dtype=torch.long)
        oracle_hops = torch.zeros(batch_size, dtype=torch.long)
        oracle_depth = torch.zeros(batch_size, dtype=torch.long)

        batch_index = torch.arange(batch_size)
        mode_one_hot = F.one_hot(modes, num_classes=self.mode_dim).float()
        payload_one_hot = F.one_hot(payloads, num_classes=self.payload_cardinality).float()
        query_one_hot = F.one_hot(queries, num_classes=self.query_cardinality).float()

        easy_mask = modes == LONG_MEMORY_MODE_EASY_EXIT
        if easy_mask.any():
            idx = torch.nonzero(easy_mask, as_tuple=False).squeeze(-1)
            labels[idx] = torch.remainder(payloads[idx], self.num_classes)
            obs[idx, 0, 0, self.mode_offset : self.mode_offset + self.mode_dim] = mode_one_hot[idx]
            obs[idx, 0, 0, self.payload_offset : self.payload_offset + self.payload_cardinality] = (
                1.5 * payload_one_hot[idx]
            )
            oracle_actions[idx, 0] = ACTION_EXIT
            oracle_action_mask[idx, 0] = 1.0

        trigger_exit_mask = modes == LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT
        if trigger_exit_mask.any():
            idx = torch.nonzero(trigger_exit_mask, as_tuple=False).squeeze(-1)
            labels[idx] = torch.remainder(payloads[idx], self.num_classes)
            obs[idx, trigger_times[idx], 0, self.trigger_offset] = 1.0
            obs[idx, trigger_times[idx], 0, self.mode_offset : self.mode_offset + self.mode_dim] = mode_one_hot[idx]
            obs[idx, trigger_times[idx], 0, self.payload_offset : self.payload_offset + self.payload_cardinality] = (
                1.5 * payload_one_hot[idx]
            )
            oracle_actions[idx, : trigger_times[idx].max().item() + 1] = ACTION_DELAY
            oracle_action_mask[idx, : trigger_times[idx].max().item() + 1] = 0.0
            for sample in idx.tolist():
                oracle_action_mask[sample, : trigger_times[sample] + 1] = 1.0
                oracle_actions[sample, trigger_times[sample]] = ACTION_EXIT
                oracle_delays[sample] = trigger_times[sample]
                oracle_exit_time[sample] = trigger_times[sample]

        final_mask = modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY
        if final_mask.any():
            idx = torch.nonzero(final_mask, as_tuple=False).squeeze(-1)
            labels[idx] = torch.remainder(payloads[idx] + queries[idx], self.num_classes)
            obs[idx, trigger_times[idx], 0, self.trigger_offset] = 1.0
            obs[idx, trigger_times[idx], 0, self.mode_offset : self.mode_offset + self.mode_dim] = mode_one_hot[idx]
            obs[idx, trigger_times[idx], 0, self.payload_offset : self.payload_offset + self.payload_cardinality] = (
                1.5 * payload_one_hot[idx]
            )
            obs[idx, self.seq_len - 1, 0, self.query_flag_offset] = 1.0
            obs[idx, self.seq_len - 1, 0, self.mode_offset : self.mode_offset + self.mode_dim] = mode_one_hot[idx]
            obs[idx, self.seq_len - 1, 0, self.query_offset : self.query_offset + self.query_cardinality] = (
                1.5 * query_one_hot[idx]
            )
            oracle_actions[idx, :] = ACTION_DELAY
            oracle_actions[idx, self.seq_len - 1] = ACTION_EXIT
            oracle_action_mask[idx, :] = 1.0
            delay_write_mask[idx, :] = 1.0
            delay_write_targets[idx, trigger_times[idx]] = 1.0
            oracle_delays[idx] = self.seq_len - 1
            oracle_exit_time[idx] = self.seq_len - 1

        retrieval_distance = torch.where(
            modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY,
            (self.seq_len - 1 - trigger_times).long(),
            torch.zeros_like(trigger_times),
        )

        return BenchmarkBatch(
            observations=obs,
            labels=labels.long(),
            modes=modes.long(),
            oracle_hops=oracle_hops,
            oracle_delays=oracle_delays,
            oracle_exit_time=oracle_exit_time,
            oracle_depth=oracle_depth,
            oracle_actions=oracle_actions,
            oracle_action_mask=oracle_action_mask,
            delay_write_targets=delay_write_targets,
            delay_write_mask=delay_write_mask,
            metadata={
                "trigger_time": trigger_times.long(),
                "query_time": torch.full_like(trigger_times, self.seq_len - 1),
                "retrieval_distance": retrieval_distance,
                "payload": payloads.long(),
                "query": queries.long(),
                "needs_final_query": (modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY).long(),
            },
        )


def build_benchmark(config: dict[str, Any]) -> SyntheticBenchmark:
    name = config["name"]
    if name == "mixed_oracle_routing":
        return MixedOracleRoutingBenchmark(config)
    if name in {"long_horizon_memory", "long_horizon_memory_v1"}:
        return LongHorizonMemoryBenchmarkV1(config)
    if name == "long_horizon_memory_v2":
        return LongHorizonMemoryBenchmarkV2(config)
    raise ValueError(f"Unknown benchmark: {name}")
