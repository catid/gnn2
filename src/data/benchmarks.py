from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


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


@dataclass
class BenchmarkBatch:
    observations: torch.Tensor
    labels: torch.Tensor
    modes: torch.Tensor
    oracle_hops: torch.Tensor
    oracle_delays: torch.Tensor
    oracle_exit_time: torch.Tensor
    oracle_depth: torch.Tensor

    def to(self, device: torch.device | str) -> "BenchmarkBatch":
        return BenchmarkBatch(
            observations=self.observations.to(device),
            labels=self.labels.to(device),
            modes=self.modes.to(device),
            oracle_hops=self.oracle_hops.to(device),
            oracle_delays=self.oracle_delays.to(device),
            oracle_exit_time=self.oracle_exit_time.to(device),
            oracle_depth=self.oracle_depth.to(device),
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
        )


class LongHorizonMemoryBenchmark(SyntheticBenchmark):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seq_len = int(config["seq_len"])
        self.payload_cardinality = int(config.get("payload_cardinality", self.num_classes))
        self.query_cardinality = int(config.get("query_cardinality", self.num_classes))
        self.trigger_offset = 0
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
        obs[:, self.seq_len - 1, 0, self.trigger_offset + 1] = 1.0
        obs[:, self.seq_len - 1, 0, self.query_offset : self.query_offset + self.query_cardinality] = (
            1.5 * query_one_hot
        )

        oracle_delays = (self.seq_len - 1 - trigger_times).long()
        oracle_exit_time = torch.full_like(oracle_delays, self.seq_len - 1)

        return BenchmarkBatch(
            observations=obs,
            labels=labels.long(),
            modes=torch.full((batch_size,), 10, dtype=torch.long),
            oracle_hops=torch.zeros(batch_size, dtype=torch.long),
            oracle_delays=oracle_delays,
            oracle_exit_time=oracle_exit_time,
            oracle_depth=torch.zeros(batch_size, dtype=torch.long),
        )


def build_benchmark(config: dict[str, Any]) -> SyntheticBenchmark:
    name = config["name"]
    if name == "mixed_oracle_routing":
        return MixedOracleRoutingBenchmark(config)
    if name == "long_horizon_memory":
        return LongHorizonMemoryBenchmark(config)
    raise ValueError(f"Unknown benchmark: {name}")
