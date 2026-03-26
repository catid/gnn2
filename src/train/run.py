from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
from typing import Any

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from src.data import BenchmarkBatch, build_benchmark
from src.data.benchmarks import (
    LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY,
    LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT,
)
from src.es import LowRankEvolutionStrategy, standardize_fitness
from src.models import ACTION_DELAY, ACTION_EXIT, PacketRoutingModel
from src.models.packet_routing import AffineAdapter, LowRankAdapter, StandardizedMLPReadoutHead
from src.utils.config import load_config


@dataclass
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


@dataclass
class TeacherDistillation:
    model: PacketRoutingModel
    route_mode: str
    temperature: float
    estimator: str
    distill_temperature: float
    target_scope: str
    logits_weight: float
    route_weight: float
    route_action_weight: float
    control_prob_weight: float
    release_prob_weight: float
    wait_prob_weight: float
    control_state_weight: float
    memory_read_weight: float
    factorized_content_weight: float
    factorized_query_weight: float
    start_step: int
    stop_step: int
    scale_start: float
    scale_end: float
    scale_schedule_steps: int
    dropout_prob_start: float
    dropout_prob_end: float
    dropout_schedule_steps: int


@dataclass
class ParameterAnchor:
    tensors: dict[str, torch.Tensor]
    weight_start: float
    weight_end: float
    weight_schedule_steps: int
    start_step: int
    stop_step: int
    p: float
    normalize: bool


@dataclass
class AuxiliaryTrainBenchmark:
    name: str
    benchmark: Any
    split: str
    batch_size: int
    loss_weight: float
    agreement_weight: float
    task_weight_cfg: dict[str, float]


@dataclass
class AuxiliaryEvalBenchmark:
    name: str
    benchmark: Any
    benchmark_name: str
    split: str
    batch_size: int
    num_batches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run packet-routing experiments.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--results-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment.seed.")
    return parser.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.seed is not None:
        experiment_cfg = cfg.setdefault("experiment", {})
        experiment_cfg["seed"] = int(args.seed)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> DistContext:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistContext(False, 0, 1, 0, device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    timeout_minutes = int(os.environ.get("GNN2_DIST_TIMEOUT_MINUTES", "60"))
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistContext(True, rank, world_size, local_rank, device)


def distributed_barrier(context: DistContext) -> None:
    if not context.enabled or not dist.is_initialized():
        return
    if context.device.type == "cuda" and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[context.local_rank])
        return
    dist.barrier()


def cleanup_distributed(context: DistContext) -> None:
    if context.enabled and dist.is_initialized():
        distributed_barrier(context)
        dist.destroy_process_group()


def create_results_dir(config: dict[str, Any], explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path

    stem = Path(config["config_path"]).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(config.get("experiment", {}).get("results_root", "results")) / f"{timestamp}_{stem}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def autocast_context(device: torch.device, enabled: bool, dtype_name: str) -> Any:
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def tensor_dict_mean(stats: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    merged: dict[str, list[torch.Tensor]] = {}
    for item in stats:
        for key, value in item.items():
            merged.setdefault(key, []).append(value.detach().float().cpu())
    return {
        key: float(torch.cat([x.reshape(-1) for x in values]).mean().item())
        for key, values in merged.items()
    }


def resolve_metric_path(metrics: dict[str, Any], path: str) -> float:
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return float("nan")
        current = current[part]
    if isinstance(current, (int, float)):
        return float(current)
    return float("nan")


def selector_score_arity(section_cfg: dict[str, Any]) -> int:
    terms_cfg = section_cfg.get("selection_metric_terms")
    if not isinstance(terms_cfg, list):
        return 1
    mode = str(section_cfg.get("selection_metric_mode", "weighted_sum"))
    if mode != "lexicographic":
        return 1
    arity = 0
    for item in terms_cfg:
        if not isinstance(item, dict) or item.get("path") is None:
            continue
        if item.get("minimum") is not None:
            arity += 1
        arity += 1
    return max(1, arity)


def initial_selection_score(section_cfg: dict[str, Any]) -> float | tuple[float, ...]:
    mode = str(section_cfg.get("selection_metric_mode", ""))
    if mode == "lexicographic":
        return tuple(float("-inf") for _ in range(selector_score_arity(section_cfg)))
    return float("-inf")


def composite_metric_score(
    section_cfg: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[float | tuple[float, ...], str] | None:
    terms_cfg = section_cfg.get("selection_metric_terms")
    if not terms_cfg:
        return None
    if not isinstance(terms_cfg, list):
        raise ValueError("selection_metric_terms must be a list of term mappings.")
    mode = str(section_cfg.get("selection_metric_mode", "weighted_sum"))
    if mode not in {"weighted_sum", "weighted_geomean", "lexicographic"}:
        raise ValueError("selection_metric_mode must be one of: weighted_sum, weighted_geomean, lexicographic")
    terms: list[tuple[str, float, float]] = []
    lexicographic_score: list[float] = []
    lexicographic_name_parts: list[str] = []
    for item in terms_cfg:
        if not isinstance(item, dict):
            raise ValueError("selection_metric_terms entries must be mappings.")
        path = item.get("path")
        if path is None:
            raise ValueError("selection_metric_terms entries require a path.")
        weight = float(item.get("weight", 1.0))
        value = resolve_metric_path(metrics, str(path))
        if math.isnan(value):
            if mode == "lexicographic":
                fallback_name_parts: list[str] = []
                for term in terms_cfg:
                    if isinstance(term, dict) and term.get("path") is not None:
                        label = str(term["path"])
                        if term.get("minimum") is not None:
                            label = f"{label}>={float(term['minimum']):g} > {label}"
                        fallback_name_parts.append(label)
                return initial_selection_score(section_cfg), f"lexicographic(" + " > ".join(fallback_name_parts) + ")"
            return float("nan"), f"{mode}(" + " + ".join(
                f"{float(term.get('weight', 1.0)):g}*{term['path']}" for term in terms_cfg if isinstance(term, dict) and "path" in term
            ) + ")"
        if mode == "lexicographic":
            minimum = item.get("minimum")
            if minimum is not None:
                minimum_value = float(minimum)
                lexicographic_score.append(1.0 if value >= minimum_value else 0.0)
                lexicographic_name_parts.append(f"{path}>={minimum_value:g}")
            lexicographic_score.append(value)
            lexicographic_name_parts.append(str(path))
            continue
        terms.append((str(path), weight, value))
    if not terms:
        if mode == "lexicographic":
            if not lexicographic_score:
                raise ValueError("selection_metric_terms must not be empty.")
            return tuple(lexicographic_score), f"lexicographic(" + " > ".join(lexicographic_name_parts) + ")"
        raise ValueError("selection_metric_terms must not be empty.")
    if mode == "weighted_sum":
        score = sum(weight * value for _, weight, value in terms)
    else:
        total_weight = sum(abs(weight) for _, weight, _ in terms)
        if total_weight <= 0.0:
            raise ValueError("weighted_geomean requires at least one non-zero weight.")
        score = math.exp(
            sum(weight * math.log(max(1e-12, value)) for _, weight, value in terms) / total_weight
        )
    metric_name = f"{mode}(" + " + ".join(f"{weight:g}*{path}" for path, weight, _ in terms) + ")"
    return score, metric_name


def evaluate_stability_guard(
    section_cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    step: int,
) -> dict[str, Any] | None:
    guard_cfg = section_cfg.get("stability_guard")
    if not isinstance(guard_cfg, dict) or not guard_cfg:
        return None
    start_step = int(guard_cfg.get("start_step", 0))
    if step < start_step:
        return None
    checks_cfg = guard_cfg.get("checks", [])
    if not isinstance(checks_cfg, list) or not checks_cfg:
        raise ValueError("training.stability_guard.checks must be a non-empty list of mappings.")
    failures: list[dict[str, float | str]] = []
    for item in checks_cfg:
        if not isinstance(item, dict):
            raise ValueError("training.stability_guard.checks entries must be mappings.")
        path = item.get("path")
        if path is None:
            raise ValueError("training.stability_guard.checks entries require a path.")
        value = resolve_metric_path(metrics, str(path))
        if math.isnan(value):
            failures.append({"path": str(path), "reason": "nan"})
            continue
        minimum = item.get("minimum")
        if minimum is not None and value < float(minimum):
            failures.append(
                {
                    "path": str(path),
                    "reason": "minimum",
                    "value": value,
                    "threshold": float(minimum),
                }
            )
        maximum = item.get("maximum")
        if maximum is not None and value > float(maximum):
            failures.append(
                {
                    "path": str(path),
                    "reason": "maximum",
                    "value": value,
                    "threshold": float(maximum),
                }
            )
    return {
        "enabled": True,
        "passed": not failures,
        "failures": failures,
        "max_consecutive_violations": int(guard_cfg.get("max_consecutive_violations", 1)),
        "max_rollbacks": int(guard_cfg.get("max_rollbacks", 0)),
        "cooldown_evals": int(guard_cfg.get("cooldown_evals", 0)),
        "early_stop_after_max_rollbacks": bool(guard_cfg.get("early_stop_after_max_rollbacks", False)),
    }


def centered_rank_fitness(rewards: torch.Tensor) -> torch.Tensor:
    rewards = rewards.float()
    order = torch.argsort(rewards)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(rewards.numel(), device=rewards.device, dtype=torch.float32)
    ranks = ranks / max(1, rewards.numel() - 1)
    return ranks - 0.5


def current_compute_penalties(
    objective_cfg: dict[str, Any],
    schedule_cfg: dict[str, Any] | None,
    step: int,
) -> dict[str, float]:
    schedule_cfg = schedule_cfg or {}
    delay_end = float(objective_cfg.get("lambda_delay", 0.0))
    delay_start = float(schedule_cfg.get("delay_penalty_start", delay_end))
    warmup_steps = int(schedule_cfg.get("delay_penalty_warmup_steps", 0))
    if warmup_steps > 0:
        frac = min(1.0, max(0.0, step / warmup_steps))
        delay_value = delay_start + frac * (delay_end - delay_start)
    else:
        delay_value = delay_end
    return {
        "hops": float(objective_cfg.get("lambda_hops", 0.0)),
        "delays": delay_value,
        "ttl_fail": float(objective_cfg.get("lambda_ttl", 0.0)),
    }


def current_training_temperature(method_cfg: dict[str, Any], step: int) -> float:
    default = float(method_cfg.get("temperature", 1.0))
    start = float(method_cfg.get("temperature_start", default))
    end = float(method_cfg.get("temperature_end", default))
    schedule_steps = int(method_cfg.get("temperature_schedule_steps", 0))
    if schedule_steps <= 0:
        return end
    frac = min(1.0, max(0.0, step / schedule_steps))
    return start + frac * (end - start)


def _scheduled_bool(routing_cfg: dict[str, Any], key: str, step: int) -> bool:
    value = bool(routing_cfg.get(key, False))
    start_step = routing_cfg.get(f"{key}_start_step")
    until_step = routing_cfg.get(f"{key}_until_step")
    if start_step is not None and step < int(start_step):
        return False
    if until_step is not None and step >= int(until_step):
        return False
    return value


def _scheduled_scalar(routing_cfg: dict[str, Any], key: str, step: int) -> float:
    if f"{key}_start" not in routing_cfg and f"{key}_end" not in routing_cfg:
        return float(routing_cfg.get(key, 0.0))
    default = float(routing_cfg.get(key, 0.0))
    start = float(routing_cfg.get(f"{key}_start", default))
    end = float(routing_cfg.get(f"{key}_end", routing_cfg.get(key, start)))
    schedule_steps = int(routing_cfg.get(f"{key}_schedule_steps", 0))
    if schedule_steps <= 0:
        return end
    frac = min(1.0, max(0.0, step / schedule_steps))
    return start + frac * (end - start)


def current_routing_cfg(routing_cfg: dict[str, Any] | None, step: int) -> dict[str, Any]:
    routing_cfg = dict(routing_cfg or {})
    boolean_keys = [
        "force_oracle_actions",
        "exit_mask_until_final",
        "exit_mask_until_trigger",
        "exit_mask_final_query_only",
        "exit_mask_trigger_exit_until_trigger",
    ]
    scalar_keys = [
        "oracle_route_weight",
        "delay_write_weight",
        "memory_payload_weight",
        "control_state_weight",
        "anti_exit_weight",
        "wait_loss_weight",
        "release_loss_weight",
    ]
    for key in boolean_keys:
        if (
            key in routing_cfg
            or f"{key}_start_step" in routing_cfg
            or f"{key}_until_step" in routing_cfg
        ):
            routing_cfg[key] = _scheduled_bool(routing_cfg, key, step)
    for key in scalar_keys:
        if (
            key in routing_cfg
            or f"{key}_start" in routing_cfg
            or f"{key}_end" in routing_cfg
            or f"{key}_schedule_steps" in routing_cfg
        ):
            routing_cfg[key] = _scheduled_scalar(routing_cfg, key, step)
    return routing_cfg


def build_routing_controls(
    batch: BenchmarkBatch,
    routing_cfg: dict[str, Any] | None,
    *,
    split: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float, float]:
    routing_cfg = routing_cfg or {}
    action_masks: torch.Tensor | None = None
    forced_actions: torch.Tensor | None = None
    oracle_actions = batch.oracle_actions
    oracle_action_mask = batch.oracle_action_mask
    oracle_route_weight = float(routing_cfg.get("oracle_route_weight", 0.0))
    delay_write_weight = float(routing_cfg.get("delay_write_weight", 0.0))

    if split == "train" and bool(routing_cfg.get("force_oracle_actions", False)) and oracle_actions is not None:
        action_mask = oracle_action_mask
        if action_mask is None:
            action_mask = torch.ones_like(oracle_actions, dtype=torch.float32)
        forced_actions = torch.where(
            action_mask > 0.0,
            oracle_actions,
            torch.full_like(oracle_actions, -1),
        )

    if split == "train" and bool(routing_cfg.get("exit_mask_until_final", False)):
        action_masks = torch.ones(
            batch.labels.shape[0],
            batch.observations.shape[1],
            3,
            device=batch.labels.device,
            dtype=batch.observations.dtype,
        )
        action_masks[:, :-1, ACTION_EXIT] = 0.0

    if split == "train" and bool(routing_cfg.get("exit_mask_until_trigger", False)) and "trigger_time" in batch.metadata:
        if action_masks is None:
            action_masks = torch.ones(
                batch.labels.shape[0],
                batch.observations.shape[1],
                3,
                device=batch.labels.device,
                dtype=batch.observations.dtype,
            )
        trigger_times = batch.metadata["trigger_time"].long()
        for sample_index in range(batch.labels.shape[0]):
            trigger_time = int(trigger_times[sample_index].item())
            if trigger_time > 0:
                action_masks[sample_index, :trigger_time, ACTION_EXIT] = 0.0

    if split == "train" and bool(routing_cfg.get("exit_mask_final_query_only", False)) and "needs_final_query" in batch.metadata:
        if action_masks is None:
            action_masks = torch.ones(
                batch.labels.shape[0],
                batch.observations.shape[1],
                3,
                device=batch.labels.device,
                dtype=batch.observations.dtype,
            )
        final_query_mask = batch.metadata["needs_final_query"].long() > 0
        if final_query_mask.any():
            action_masks[final_query_mask, :-1, ACTION_EXIT] = 0.0

    if split == "train" and bool(routing_cfg.get("exit_mask_trigger_exit_until_trigger", False)) and "trigger_time" in batch.metadata:
        if action_masks is None:
            action_masks = torch.ones(
                batch.labels.shape[0],
                batch.observations.shape[1],
                3,
                device=batch.labels.device,
                dtype=batch.observations.dtype,
            )
        trigger_mask = batch.modes == LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT
        if trigger_mask.any():
            trigger_times = batch.metadata["trigger_time"].long()
            for sample_index in torch.nonzero(trigger_mask, as_tuple=False).squeeze(-1).tolist():
                trigger_time = int(trigger_times[sample_index].item())
                if trigger_time > 0:
                    action_masks[sample_index, :trigger_time, ACTION_EXIT] = 0.0

    return (
        forced_actions,
        action_masks,
        oracle_actions,
        oracle_action_mask,
        oracle_route_weight,
        delay_write_weight,
    )


def build_memory_controls(
    batch: BenchmarkBatch,
    routing_cfg: dict[str, Any] | None,
    *,
    split: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    routing_cfg = routing_cfg or {}
    memory_payload_weight = float(routing_cfg.get("memory_payload_weight", 0.0))
    if split != "train" or memory_payload_weight <= 0.0 or "query_time" not in batch.metadata or "payload" not in batch.metadata:
        return None, None, 0.0

    batch_size = batch.labels.shape[0]
    seq_len = batch.observations.shape[1]
    device = batch.labels.device
    mask = torch.zeros(batch_size, seq_len, device=device, dtype=batch.observations.dtype)
    targets = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
    query_times = batch.metadata["query_time"].long()
    payloads = batch.metadata["payload"].long()

    sample_mask = torch.ones(batch_size, device=device, dtype=batch.observations.dtype)
    if "needs_final_query" in batch.metadata:
        sample_mask = batch.metadata["needs_final_query"].to(device=device, dtype=batch.observations.dtype)

    batch_index = torch.arange(batch_size, device=device)
    targets[batch_index, query_times] = payloads
    mask[batch_index, query_times] = sample_mask
    return targets, mask, memory_payload_weight


def build_control_controls(
    batch: BenchmarkBatch,
    routing_cfg: dict[str, Any] | None,
    *,
    split: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float, torch.Tensor | None, float]:
    routing_cfg = routing_cfg or {}
    control_weight = float(routing_cfg.get("control_state_weight", 0.0))
    anti_exit_weight = float(routing_cfg.get("anti_exit_weight", 0.0))
    if (
        split != "train"
        or (control_weight <= 0.0 and anti_exit_weight <= 0.0)
        or "trigger_time" not in batch.metadata
        or "query_time" not in batch.metadata
        or "needs_final_query" not in batch.metadata
    ):
        return None, None, 0.0, None, 0.0

    batch_size = batch.labels.shape[0]
    seq_len = batch.observations.shape[1]
    device = batch.labels.device
    dtype = batch.observations.dtype
    targets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    target_mask = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    anti_exit_mask = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    target_scope = str(routing_cfg.get("control_target_scope", "final_query_inclusive"))

    trigger_times = batch.metadata["trigger_time"].long()
    query_times = batch.metadata["query_time"].long()
    final_query_mask = batch.metadata["needs_final_query"].long() > 0

    if target_scope == "oracle_all":
        if batch.oracle_actions is None or batch.oracle_action_mask is None:
            return None, None, 0.0, None, 0.0
        targets = (batch.oracle_actions == ACTION_DELAY).to(device=device, dtype=dtype)
        target_mask = batch.oracle_action_mask.to(device=device, dtype=dtype)
        anti_exit_mask = targets.clone()
        return targets, target_mask, control_weight, anti_exit_mask, anti_exit_weight
    if target_scope == "oracle_delayed_only":
        if batch.oracle_actions is None or batch.oracle_action_mask is None:
            return None, None, 0.0, None, 0.0
        delayed_modes = (
            (batch.modes == LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT)
            | (batch.modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY)
        ).to(device=device, dtype=dtype)
        targets = (batch.oracle_actions == ACTION_DELAY).to(device=device, dtype=dtype)
        target_mask = batch.oracle_action_mask.to(device=device, dtype=dtype) * delayed_modes.unsqueeze(1)
        anti_exit_mask = targets * delayed_modes.unsqueeze(1)
        return targets, target_mask, control_weight, anti_exit_mask, anti_exit_weight
    if target_scope not in {"final_query_inclusive", "final_query_wait_only"}:
        raise ValueError(f"Unknown control_target_scope: {target_scope}")

    for sample_index in torch.nonzero(final_query_mask, as_tuple=False).squeeze(-1).tolist():
        trigger_time = int(trigger_times[sample_index].item())
        query_time = int(query_times[sample_index].item())
        trigger_time = max(0, min(trigger_time, seq_len - 1))
        query_time = max(trigger_time, min(query_time, seq_len - 1))
        target_stop = query_time + 1 if target_scope == "final_query_inclusive" else query_time
        if target_stop > trigger_time:
            targets[sample_index, trigger_time:target_stop] = 1.0
        target_mask[sample_index, trigger_time : query_time + 1] = 1.0
        anti_exit_mask[sample_index, trigger_time:query_time] = 1.0

    return targets, target_mask, control_weight, anti_exit_mask, anti_exit_weight


def build_wait_controls(
    batch: BenchmarkBatch,
    routing_cfg: dict[str, Any] | None,
    *,
    split: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float, float, float]:
    routing_cfg = routing_cfg or {}
    wait_weight = float(routing_cfg.get("wait_loss_weight", 0.0))
    wait_positive_weight = float(routing_cfg.get("wait_positive_weight", 1.0))
    wait_negative_weight = float(routing_cfg.get("wait_negative_weight", 1.0))
    if (
        split != "train"
        or wait_weight <= 0.0
        or "trigger_time" not in batch.metadata
        or "query_time" not in batch.metadata
        or "needs_final_query" not in batch.metadata
    ):
        return None, None, 0.0, 1.0, 1.0

    batch_size = batch.labels.shape[0]
    seq_len = batch.observations.shape[1]
    device = batch.labels.device
    dtype = batch.observations.dtype
    targets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    target_mask = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    target_scope = str(routing_cfg.get("wait_target_scope", "final_query_only"))

    if target_scope == "oracle_all":
        if batch.oracle_actions is None or batch.oracle_action_mask is None:
            return None, None, 0.0, 1.0, 1.0
        targets = (batch.oracle_actions == ACTION_DELAY).to(dtype=dtype)
        target_mask = batch.oracle_action_mask.to(device=device, dtype=dtype)
        return targets, target_mask, wait_weight, wait_positive_weight, wait_negative_weight
    if target_scope == "oracle_delayed_only":
        if batch.oracle_actions is None or batch.oracle_action_mask is None:
            return None, None, 0.0, 1.0, 1.0
        delayed_modes = (
            (batch.modes == LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT)
            | (batch.modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY)
        ).to(device=device, dtype=dtype)
        targets = (batch.oracle_actions == ACTION_DELAY).to(dtype=dtype)
        target_mask = batch.oracle_action_mask.to(device=device, dtype=dtype) * delayed_modes.unsqueeze(1)
        return targets, target_mask, wait_weight, wait_positive_weight, wait_negative_weight
    if target_scope != "final_query_only":
        raise ValueError(f"Unknown wait_target_scope: {target_scope}")

    trigger_times = batch.metadata["trigger_time"].long()
    query_times = batch.metadata["query_time"].long()
    final_query_mask = batch.metadata["needs_final_query"].long() > 0

    for sample_index in torch.nonzero(final_query_mask, as_tuple=False).squeeze(-1).tolist():
        trigger_time = int(trigger_times[sample_index].item())
        query_time = int(query_times[sample_index].item())
        trigger_time = max(0, min(trigger_time, seq_len - 1))
        query_time = max(trigger_time, min(query_time, seq_len - 1))
        targets[sample_index, trigger_time:query_time] = 1.0
        target_mask[sample_index, trigger_time : query_time + 1] = 1.0

    return targets, target_mask, wait_weight, wait_positive_weight, wait_negative_weight


def build_release_controls(
    batch: BenchmarkBatch,
    routing_cfg: dict[str, Any] | None,
    *,
    split: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float, float]:
    routing_cfg = routing_cfg or {}
    release_weight = float(routing_cfg.get("release_loss_weight", 0.0))
    if split != "train" or release_weight <= 0.0 or batch.oracle_actions is None or batch.oracle_action_mask is None:
        return None, None, 0.0, 1.0

    dtype = batch.observations.dtype
    device = batch.labels.device
    target_scope = str(routing_cfg.get("release_target_scope", "oracle_all"))
    targets = (batch.oracle_actions != ACTION_DELAY).to(device=device, dtype=dtype)
    target_mask = batch.oracle_action_mask.to(device=device, dtype=dtype)
    if target_scope == "final_query_only":
        final_query_mask = batch.metadata.get("needs_final_query")
        if final_query_mask is None:
            return None, None, 0.0, 1.0
        target_mask = target_mask * final_query_mask.to(device=device, dtype=dtype).unsqueeze(1)
    elif target_scope == "delayed_only":
        delayed_mask = (
            (batch.modes == LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT)
            | (batch.modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY)
        ).to(device=device, dtype=dtype)
        target_mask = target_mask * delayed_mask.unsqueeze(1)
    elif target_scope != "oracle_all":
        raise ValueError(f"Unknown release_target_scope: {target_scope}")

    release_positive_weight = float(routing_cfg.get("release_positive_weight", 1.0))
    return targets, target_mask, release_weight, release_positive_weight


def build_task_sample_weights(
    batch: BenchmarkBatch,
    train_cfg: dict[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    train_cfg = train_cfg or {}
    final_query_weight = float(train_cfg.get("final_query_weight", 1.0))
    non_final_query_weight = float(train_cfg.get("non_final_query_weight", 1.0))
    if final_query_weight == 1.0 and non_final_query_weight == 1.0:
        return None
    sample_weights = torch.full(
        (batch.labels.shape[0],),
        non_final_query_weight,
        device=device,
        dtype=dtype,
    )
    final_query_mask = batch.metadata.get("needs_final_query")
    if final_query_mask is not None:
        sample_weights = torch.where(
            final_query_mask.to(device=device, dtype=torch.bool),
            torch.full_like(sample_weights, final_query_weight),
            sample_weights,
        )
    return sample_weights


def load_auxiliary_train_benchmarks(cfg: dict[str, Any]) -> list[AuxiliaryTrainBenchmark]:
    train_cfg = cfg.get("training", {})
    specs = train_cfg.get("auxiliary_train_benchmarks", [])
    if not specs:
        return []

    config_base = Path(str(cfg.get("config_path", "."))).resolve().parent
    default_task_weight_cfg = {
        "final_query_weight": float(train_cfg.get("final_query_weight", 1.0)),
        "non_final_query_weight": float(train_cfg.get("non_final_query_weight", 1.0)),
    }

    sources: list[AuxiliaryTrainBenchmark] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError("training.auxiliary_train_benchmarks entries must be mappings.")
        config_ref = spec.get("config")
        if not config_ref:
            raise ValueError("training.auxiliary_train_benchmarks requires a config field per entry.")
        config_path = Path(str(config_ref))
        if not config_path.is_absolute():
            config_path = (config_base / config_path).resolve()
        aux_cfg = load_config(config_path)
        benchmark_cfg = aux_cfg.get("benchmark")
        if not isinstance(benchmark_cfg, dict):
            raise ValueError(f"Auxiliary benchmark config {config_path} must define a benchmark section.")

        task_weight_cfg = dict(default_task_weight_cfg)
        if "final_query_weight" in spec:
            task_weight_cfg["final_query_weight"] = float(spec["final_query_weight"])
        if "non_final_query_weight" in spec:
            task_weight_cfg["non_final_query_weight"] = float(spec["non_final_query_weight"])

        sources.append(
            AuxiliaryTrainBenchmark(
                name=str(spec.get("name", f"aux_{index}")),
                benchmark=build_benchmark(benchmark_cfg),
                split=str(spec.get("split", "confirm")),
                batch_size=int(spec.get("batch_size", train_cfg["batch_size"])),
                loss_weight=float(spec.get("loss_weight", 1.0)),
                agreement_weight=float(spec.get("agreement_weight", 0.0)),
                task_weight_cfg=task_weight_cfg,
            )
        )
    return sources


def load_auxiliary_eval_benchmarks(
    cfg: dict[str, Any],
    *,
    section: str = "training",
) -> list[AuxiliaryEvalBenchmark]:
    section_cfg = cfg.get(section, {})
    train_cfg = cfg.get("training", {})
    specs = section_cfg.get("selection_eval_benchmarks", [])
    if not specs:
        return []

    config_base = Path(str(cfg.get("config_path", "."))).resolve().parent
    default_batch_size = int(
        section_cfg.get(
            "eval_batch_size",
            section_cfg.get(
                "val_batch_size",
                train_cfg.get("val_batch_size", train_cfg.get("batch_size", 1)),
            ),
        )
    )
    default_num_batches = int(section_cfg.get("val_batches", train_cfg.get("val_batches", 8)))

    sources: list[AuxiliaryEvalBenchmark] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"{section}.selection_eval_benchmarks entries must be mappings.")
        config_ref = spec.get("config")
        if not config_ref:
            raise ValueError(f"{section}.selection_eval_benchmarks requires a config field per entry.")
        config_path = Path(str(config_ref))
        if not config_path.is_absolute():
            config_path = (config_base / config_path).resolve()
        aux_cfg = load_config(config_path)
        benchmark_cfg = aux_cfg.get("benchmark")
        if not isinstance(benchmark_cfg, dict):
            raise ValueError(f"Selection eval benchmark config {config_path} must define a benchmark section.")

        benchmark_name = str(benchmark_cfg.get("name", ""))
        if not benchmark_name:
            raise ValueError(f"Selection eval benchmark config {config_path} must define benchmark.name.")

        sources.append(
            AuxiliaryEvalBenchmark(
                name=str(spec.get("name", f"proxy_{index}")),
                benchmark=build_benchmark(benchmark_cfg),
                benchmark_name=benchmark_name,
                split=str(spec.get("split", "confirm")),
                batch_size=int(spec.get("batch_size", default_batch_size)),
                num_batches=int(spec.get("num_batches", default_num_batches)),
            )
        )
    return sources


def build_reward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, torch.Tensor],
    objective_cfg: dict[str, Any],
    reward_penalties: dict[str, float] | None = None,
) -> torch.Tensor:
    task_metric = objective_cfg.get("task_score", "neg_ce")
    if task_metric == "accuracy":
        task_score = (logits.argmax(dim=-1) == labels).float()
    else:
        task_score = -F.cross_entropy(logits, labels, reduction="none")

    penalties = reward_penalties or {
        "hops": float(objective_cfg.get("lambda_hops", 0.0)),
        "delays": float(objective_cfg.get("lambda_delay", 0.0)),
        "ttl_fail": float(objective_cfg.get("lambda_ttl", 0.0)),
    }
    reward = task_score.clone()
    reward = reward - float(penalties.get("hops", 0.0)) * stats["hops"]
    reward = reward - float(penalties.get("delays", 0.0)) * stats["delays"]
    reward = reward - float(penalties.get("ttl_fail", 0.0)) * stats["ttl_fail"]
    return reward


def summarize_batch(
    batch: BenchmarkBatch,
    output,
    benchmark_name: str,
) -> dict[str, torch.Tensor]:
    predictions = output.logits.argmax(dim=-1)
    rounded_hops = torch.round(output.stats["hops"]).long()
    rounded_delays = torch.round(output.stats["delays"]).long()
    rounded_exit = torch.round(output.stats["exit_time"]).long()
    route_match = (
        (rounded_hops == batch.oracle_hops)
        & (rounded_delays == batch.oracle_delays)
        & (rounded_exit == batch.oracle_exit_time)
    ).float()

    metrics = {
        "accuracy": (predictions == batch.labels).float(),
        "task_loss": torch.full_like(output.stats["accuracy"], float(output.task_loss.detach().item())),
        "loss": torch.full_like(output.stats["accuracy"], float(output.loss.detach().item())),
        "route_loss": torch.full_like(output.stats["accuracy"], float(output.route_loss.detach().item())),
    }
    metrics.update(output.stats)
    if benchmark_name.startswith("long_horizon_memory") or benchmark_name == "mixed_oracle_routing":
        metrics["route_match"] = route_match
    if benchmark_name.startswith("long_horizon_memory") and "query_time" in batch.metadata:
        query_time = batch.metadata["query_time"].float()
        final_mask = (
            batch.metadata.get("needs_final_query", torch.zeros_like(query_time)).float()
        )
        premature_exit = ((output.stats["exit_time"] < query_time).float()) * final_mask
        metrics["premature_exit_rate"] = premature_exit
        metrics["final_query_wait_gap"] = (query_time - output.stats["exit_time"]).clamp_min(0.0) * final_mask
    metrics["early_exit_rate"] = output.stats["early_exit_mass"]
    return metrics


def grouped_mode_metrics(
    benchmark,
    batch: BenchmarkBatch,
    metrics: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    mode_names = getattr(benchmark, "mode_names", {}) or {}
    if not mode_names:
        return {}

    grouped: dict[str, dict[str, float]] = {}
    tracked_keys = sorted(metrics.keys())
    for mode_id, mode_name in mode_names.items():
        mask = batch.modes == mode_id
        count = int(mask.sum().item())
        if count == 0:
            continue
        grouped[mode_name] = {"count": float(count)}
        for key in tracked_keys:
            grouped[mode_name][key] = float(metrics[key][mask].float().mean().item())
    return grouped


def grouped_slice_metrics(
    benchmark_name: str,
    batch: BenchmarkBatch,
    metrics: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    if not benchmark_name.startswith("long_horizon_memory"):
        return {}
    if not batch.metadata:
        return {}

    seq_len = batch.observations.shape[1]
    tracked_keys = [
        "accuracy",
        "compute",
        "delay_rate",
        "delays",
        "delay_retain_mean",
        "delay_write_mean",
        "memory_read_mean",
        "memory_write_mean",
        "memory_read_entropy",
        "memory_write_entropy",
        "early_exit_rate",
        "exit_rate",
        "exit_time",
        "packet_age_mean",
        "route_entropy",
        "router_confidence",
        "route_match",
        "premature_exit_rate",
        "final_query_wait_gap",
        "control_state_mean",
        "control_prob_mean",
        "control_set_mean",
        "control_clear_mean",
        "wait_state_mean",
        "wait_prob_mean",
        "release_prob_mean",
        "control_loss",
        "wait_loss",
        "release_loss",
        "anti_exit_loss",
    ]
    grouped: dict[str, dict[str, float]] = {}

    trigger_times = batch.metadata.get("trigger_time")
    if trigger_times is not None:
        trigger_masks = {
            "trigger_early": trigger_times < max(1, seq_len // 3),
            "trigger_mid": (trigger_times >= max(1, seq_len // 3)) & (trigger_times < max(1, 2 * seq_len // 3)),
            "trigger_late": trigger_times >= max(1, 2 * seq_len // 3),
        }
        grouped.update(_masked_group_metrics(trigger_masks, metrics, tracked_keys))

    retrieval_distance = batch.metadata.get("retrieval_distance")
    if retrieval_distance is not None:
        distance_masks = {
            "distance_short": retrieval_distance < max(1, seq_len // 4),
            "distance_mid": (retrieval_distance >= max(1, seq_len // 4)) & (retrieval_distance < max(1, seq_len // 2)),
            "distance_long": retrieval_distance >= max(1, seq_len // 2),
        }
        grouped.update(_masked_group_metrics(distance_masks, metrics, tracked_keys))

    return grouped


def _masked_group_metrics(
    masks: dict[str, torch.Tensor],
    metrics: dict[str, torch.Tensor],
    tracked_keys: list[str],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for group_name, mask in masks.items():
        count = int(mask.sum().item())
        if count == 0:
            continue
        grouped[group_name] = {"count": float(count)}
        for key in tracked_keys:
            if key in metrics:
                grouped[group_name][key] = float(metrics[key][mask].float().mean().item())
    return grouped


def evaluate_model(
    model: PacketRoutingModel,
    benchmark,
    device: torch.device,
    benchmark_name: str,
    split: str,
    num_batches: int,
    batch_size: int,
    route_mode: str,
    compute_penalties: dict[str, float],
    temperature: float,
    estimator: str,
    amp_enabled: bool,
    amp_dtype: str,
    routing_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    start_time = time.time()
    collected: list[dict[str, torch.Tensor]] = []
    mode_sums: dict[str, dict[str, float]] = {}
    slice_sums: dict[str, dict[str, float]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for step in range(num_batches):
            batch = benchmark.sample_batch(batch_size=batch_size, split=split, step=step, device=device)
            forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
                batch,
                routing_cfg,
                split=split,
            )
            memory_payload_targets, memory_payload_mask, memory_payload_weight = build_memory_controls(
                batch,
                routing_cfg,
                split=split,
            )
            control_targets, control_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
                batch,
                routing_cfg,
                split=split,
            )
            wait_targets, wait_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
                batch,
                routing_cfg,
                split=split,
            )
            release_targets, release_mask, release_weight, release_positive_weight = build_release_controls(
                batch,
                routing_cfg,
                split=split,
            )
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model(
                    observations=batch.observations,
                    labels=batch.labels,
                    route_mode=route_mode,
                    compute_penalties=compute_penalties,
                    temperature=temperature,
                    estimator=estimator,
                    truncate_bptt_steps=0,
                    forced_actions=forced_actions,
                    action_masks=action_masks,
                    oracle_actions=oracle_actions,
                    oracle_action_mask=oracle_action_mask,
                    oracle_route_weight=oracle_route_weight,
                    delay_write_targets=batch.delay_write_targets,
                    delay_write_mask=batch.delay_write_mask,
                    delay_write_weight=delay_write_weight,
                    memory_payload_targets=memory_payload_targets,
                    memory_payload_mask=memory_payload_mask,
                    memory_payload_weight=memory_payload_weight,
                    control_targets=control_targets,
                    control_mask=control_mask,
                    control_weight=control_weight,
                    anti_exit_mask=anti_exit_mask,
                    anti_exit_weight=anti_exit_weight,
                    wait_targets=wait_targets,
                    wait_mask=wait_mask,
                    wait_weight=wait_weight,
                    wait_positive_weight=wait_positive_weight,
                    wait_negative_weight=wait_negative_weight,
                    release_targets=release_targets,
                    release_mask=release_mask,
                    release_weight=release_weight,
                    release_positive_weight=release_positive_weight,
                    final_query_mask=batch.metadata.get("needs_final_query"),
                )
            batch_metrics = summarize_batch(batch, output, benchmark_name)
            collected.append(batch_metrics)
            for mode_name, mode_metrics in grouped_mode_metrics(benchmark, batch, batch_metrics).items():
                target = mode_sums.setdefault(mode_name, {})
                count = float(mode_metrics["count"])
                target["count"] = target.get("count", 0.0) + count
                for key, value in mode_metrics.items():
                    if key == "count":
                        continue
                    target[key] = target.get(key, 0.0) + value * count
            for slice_name, slice_metrics in grouped_slice_metrics(benchmark_name, batch, batch_metrics).items():
                target = slice_sums.setdefault(slice_name, {})
                count = float(slice_metrics["count"])
                target["count"] = target.get("count", 0.0) + count
                for key, value in slice_metrics.items():
                    if key == "count":
                        continue
                    target[key] = target.get(key, 0.0) + value * count
    metrics = tensor_dict_mean(collected)
    duration = max(1e-6, time.time() - start_time)
    if mode_sums:
        metrics["per_mode"] = {
            mode_name: {
                key: (value / mode_metrics["count"] if key != "count" else value)
                for key, value in mode_metrics.items()
            }
            for mode_name, mode_metrics in mode_sums.items()
        }
        for mode_name, mode_metrics in metrics["per_mode"].items():
            mode_metrics["count"] = int(mode_metrics["count"])
    if slice_sums:
        metrics["per_slice"] = {
            slice_name: {
                key: (value / slice_metrics["count"] if key != "count" else value)
                for key, value in slice_metrics.items()
            }
            for slice_name, slice_metrics in slice_sums.items()
        }
        for slice_name, slice_metrics in metrics["per_slice"].items():
            slice_metrics["count"] = int(slice_metrics["count"])
    metrics["examples_seen"] = batch_size * num_batches
    metrics["examples_per_sec"] = batch_size * num_batches / duration
    metrics["wall_time_sec"] = duration
    metrics["peak_memory_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    return metrics


def validation_score(
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    section: str = "training",
) -> tuple[float | tuple[float, ...], str]:
    section_cfg = cfg.get(section, {})
    composite = composite_metric_score(section_cfg, metrics)
    if composite is not None:
        return composite
    metric_path = str(section_cfg.get("selection_metric", "accuracy"))
    score = resolve_metric_path(metrics, metric_path)
    return score, metric_path


def evaluation_requests(
    *,
    section_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[int, list[tuple[str, int]]]:
    eval_batch_size = int(
        section_cfg.get(
            "eval_batch_size",
            section_cfg.get(
                "val_batch_size",
                train_cfg.get("val_batch_size", train_cfg["batch_size"]),
            ),
        )
    )
    requests = [
        ("test", int(section_cfg.get("test_batches", train_cfg.get("test_batches", 16)))),
    ]
    confirm_batches = int(section_cfg.get("confirm_batches", train_cfg.get("confirm_batches", 0)))
    if confirm_batches > 0:
        requests.append(("confirm", confirm_batches))
    return eval_batch_size, requests


def save_checkpoint(path: Path, model: PacketRoutingModel, optimizer: torch.optim.Optimizer | None, step: int, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def configure_trainable_parameters(
    model: PacketRoutingModel,
    section_cfg: dict[str, Any],
) -> dict[str, Any]:
    trainable_prefixes = tuple(section_cfg.get("trainable_prefixes", []))
    freeze_prefixes = tuple(section_cfg.get("freeze_prefixes", []))
    trainable_exact = set(section_cfg.get("trainable_exact_names", []))
    freeze_exact = set(section_cfg.get("freeze_exact_names", []))

    def matches(name: str, prefixes: tuple[str, ...], exact: set[str]) -> bool:
        return name in exact or any(name.startswith(prefix) for prefix in prefixes)

    use_allowlist = bool(trainable_prefixes or trainable_exact)
    enabled_names: list[str] = []
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        total_params += parameter.numel()
        allow = matches(name, trainable_prefixes, trainable_exact) if use_allowlist else True
        blocked = matches(name, freeze_prefixes, freeze_exact)
        parameter.requires_grad = allow and not blocked
        if parameter.requires_grad:
            enabled_names.append(name)
            trainable_params += parameter.numel()
    return {
        "total_parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "trainable_parameter_names": enabled_names,
    }


def configure_es_parameter_names(
    model: PacketRoutingModel,
    es_cfg: dict[str, Any],
) -> dict[str, Any]:
    custom_filters = any(
        es_cfg.get(key)
        for key in (
            "trainable_prefixes",
            "freeze_prefixes",
            "trainable_exact_names",
            "freeze_exact_names",
        )
    )
    if custom_filters:
        info = configure_trainable_parameters(model, es_cfg)
        return {
            **info,
            "es_parameter_names": list(info["trainable_parameter_names"]),
        }

    parameter_names = model.es_parameter_names(include_adapters=bool(es_cfg.get("evolve_adapters", False)))
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        total_params += parameter.numel()
        parameter.requires_grad = name in parameter_names
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    return {
        "total_parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "trainable_parameter_names": list(parameter_names),
        "es_parameter_names": list(parameter_names),
    }


def apply_partial_init(
    model: PacketRoutingModel,
    init_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if not init_cfg:
        return {}

    source_cfgs = list(init_cfg.get("sources", []) or [])
    source_filter_cfgs: list[dict[str, Any]] = []
    if source_cfgs:
        resolved_checkpoints: list[Path] = []
        source_states: list[dict[str, Any]] = []
        source_weights: list[float] = []
        for source_cfg in source_cfgs:
            run_dir_value = source_cfg.get("run_dir")
            if not run_dir_value:
                raise ValueError("partial_init.sources[].run_dir is required when partial_init.sources is configured.")
            checkpoint_path = resolve_checkpoint_from_run_dir(Path(run_dir_value), source_cfg.get("checkpoint"))
            payload = torch.load(checkpoint_path, map_location="cpu")
            resolved_checkpoints.append(checkpoint_path)
            source_states.append(payload["model"])
            source_weights.append(float(source_cfg.get("weight", 1.0)))
            source_filter_cfgs.append(
                {
                    "include_prefixes": tuple(source_cfg.get("include_prefixes", [])),
                    "exclude_prefixes": tuple(source_cfg.get("exclude_prefixes", [])),
                    "include_exact_names": set(source_cfg.get("include_exact_names", [])),
                    "exclude_exact_names": set(source_cfg.get("exclude_exact_names", [])),
                }
            )
        total_weight = sum(source_weights)
        if total_weight <= 0.0:
            raise ValueError("partial_init.sources weights must sum to a positive value.")
        source_weights = [weight / total_weight for weight in source_weights]
    else:
        run_dir_value = init_cfg.get("run_dir")
        if not run_dir_value:
            raise ValueError("partial_init.run_dir is required when partial_init is configured.")
        checkpoint_path = resolve_checkpoint_from_run_dir(Path(run_dir_value), init_cfg.get("checkpoint"))
        payload = torch.load(checkpoint_path, map_location="cpu")
        resolved_checkpoints = [checkpoint_path]
        source_states = [payload["model"]]
        source_weights = [1.0]
        source_filter_cfgs = [
            {
                "include_prefixes": tuple(),
                "exclude_prefixes": tuple(),
                "include_exact_names": set(),
                "exclude_exact_names": set(),
            }
        ]

    current_state = model.state_dict()

    include_prefixes = tuple(init_cfg.get("include_prefixes", []))
    exclude_prefixes = tuple(init_cfg.get("exclude_prefixes", []))
    include_exact = set(init_cfg.get("include_exact_names", []))
    exclude_exact = set(init_cfg.get("exclude_exact_names", []))

    def matches(name: str, prefixes: tuple[str, ...], exact: set[str]) -> bool:
        return name in exact or any(name.startswith(prefix) for prefix in prefixes)

    use_allowlist = bool(include_prefixes or include_exact)
    copied_names: list[str] = []
    skipped_shape: list[str] = []
    skipped_filter: list[str] = []
    for name in source_states[0]:
        if name not in current_state:
            skipped_shape.append(name)
            continue
        tensors: list[torch.Tensor] = []
        expected_shape = current_state[name].shape
        shape_mismatch = False
        for source_state in source_states:
            tensor = source_state.get(name)
            if tensor is None or tensor.shape != expected_shape:
                shape_mismatch = True
                break
            tensors.append(tensor)
        if shape_mismatch:
            skipped_shape.append(name)
            continue
        allow = matches(name, include_prefixes, include_exact) if use_allowlist else True
        blocked = matches(name, exclude_prefixes, exclude_exact)
        if not allow or blocked:
            skipped_filter.append(name)
            continue
        blended = torch.zeros_like(current_state[name])
        participating: list[tuple[torch.Tensor, float]] = []
        for tensor, weight, filter_cfg in zip(tensors, source_weights, source_filter_cfgs):
            source_allow_prefixes = filter_cfg["include_prefixes"]
            source_allow_exact = filter_cfg["include_exact_names"]
            source_block_prefixes = filter_cfg["exclude_prefixes"]
            source_block_exact = filter_cfg["exclude_exact_names"]
            source_use_allowlist = bool(source_allow_prefixes or source_allow_exact)
            source_allow = matches(name, source_allow_prefixes, source_allow_exact) if source_use_allowlist else True
            source_blocked = matches(name, source_block_prefixes, source_block_exact)
            if source_allow and not source_blocked:
                participating.append((tensor, weight))
        if not participating:
            skipped_filter.append(name)
            continue
        active_weight = sum(weight for _, weight in participating)
        for tensor, weight in participating:
            blended.add_(tensor.to(device=blended.device, dtype=blended.dtype), alpha=weight)
        if active_weight != 1.0:
            blended.div_(active_weight)
        current_state[name].copy_(blended)
        copied_names.append(name)

    if not copied_names:
        joined = ", ".join(str(path) for path in resolved_checkpoints)
        raise ValueError(f"partial_init copied zero parameters from {joined}.")

    model.load_state_dict(current_state, strict=False)
    summary = {
        "partial_init_checkpoint": str(resolved_checkpoints[0]),
        "partial_init_parameter_count": len(copied_names),
        "partial_init_parameter_names": copied_names,
        "partial_init_skipped_shape_count": len(skipped_shape),
        "partial_init_skipped_filter_count": len(skipped_filter),
    }
    if len(resolved_checkpoints) > 1:
        summary["partial_init_checkpoint"] = ",".join(str(path) for path in resolved_checkpoints)
        summary["partial_init_checkpoints"] = [str(path) for path in resolved_checkpoints]
        summary["partial_init_weights"] = list(source_weights)
    return summary


def apply_probe_warmstart(
    model: PacketRoutingModel,
    *,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    route_mode: str,
    temperature: float,
    estimator: str,
) -> dict[str, Any]:
    return _apply_probe_head_or_adapter_warmstart(
        model,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        cfg=cfg,
        device=device,
        route_mode=route_mode,
        temperature=temperature,
        estimator=estimator,
        target_kind="readout_head",
    )


def _collect_probe_dataset(
    model: PacketRoutingModel,
    *,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    route_mode: str,
    temperature: float,
    estimator: str,
    split: str,
    num_batches: int,
    batch_size: int,
    final_query_only: bool,
    feature_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    amp_enabled = bool(cfg.get("system", {}).get("amp", False))
    amp_dtype = str(cfg.get("system", {}).get("amp_dtype", "bf16"))
    compute_penalties = {"hops": 0.0, "delays": 0.0, "ttl_fail": 0.0}
    routing_cfg = cfg.get("routing")

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for step in range(num_batches):
            batch = benchmark.sample_batch(
                batch_size=batch_size,
                split=split,
                step=step,
                device=device,
            )
            forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
                batch,
                routing_cfg,
                split=split,
            )
            memory_payload_targets, memory_payload_mask, memory_payload_weight = build_memory_controls(
                batch,
                routing_cfg,
                split=split,
            )
            control_targets, control_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
                batch,
                routing_cfg,
                split=split,
            )
            wait_targets, wait_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
                batch,
                routing_cfg,
                split=split,
            )
            release_targets, release_mask, release_weight, release_positive_weight = build_release_controls(
                batch,
                routing_cfg,
                split=split,
            )
            with autocast_context(device, amp_enabled, amp_dtype):
                output = model(
                    observations=batch.observations,
                    labels=batch.labels,
                    route_mode=route_mode,
                    compute_penalties=compute_penalties,
                    temperature=temperature,
                    estimator=estimator,
                    truncate_bptt_steps=0,
                    forced_actions=forced_actions,
                    action_masks=action_masks,
                    oracle_actions=oracle_actions,
                    oracle_action_mask=oracle_action_mask,
                    oracle_route_weight=oracle_route_weight,
                    delay_write_targets=batch.delay_write_targets,
                    delay_write_mask=batch.delay_write_mask,
                    delay_write_weight=delay_write_weight,
                    memory_payload_targets=memory_payload_targets,
                    memory_payload_mask=memory_payload_mask,
                    memory_payload_weight=memory_payload_weight,
                    control_targets=control_targets,
                    control_mask=control_mask,
                    control_weight=control_weight,
                    anti_exit_mask=anti_exit_mask,
                    anti_exit_weight=anti_exit_weight,
                    wait_targets=wait_targets,
                    wait_mask=wait_mask,
                    wait_weight=wait_weight,
                    wait_positive_weight=wait_positive_weight,
                    wait_negative_weight=wait_negative_weight,
                    release_targets=release_targets,
                    release_mask=release_mask,
                    release_weight=release_weight,
                    release_positive_weight=release_positive_weight,
                    factorized_payload_targets=batch.metadata.get("payload"),
                    factorized_query_targets=batch.metadata.get("query"),
                    final_query_mask=batch.metadata.get("needs_final_query"),
                    return_trace=True,
                )
            feature_tensor = (output.trace or {}).get(feature_source)
            if feature_tensor is None:
                continue
            if final_query_only:
                mask = batch.metadata.get("needs_final_query")
                if mask is None:
                    continue
                mask = mask > 0
            else:
                mask = torch.ones_like(batch.labels, dtype=torch.bool)
            if not bool(mask.any().item()):
                continue
            features.append(feature_tensor[mask].detach())
            labels.append(batch.labels[mask].detach())
    if was_training:
        model.train()

    if not features:
        raise ValueError(
            f"probe warmstart collected no samples for source={feature_source} "
            f"on {benchmark_name} split={split}."
        )

    train_x = torch.cat(features, dim=0).to(device=device, dtype=torch.float32)
    train_y = torch.cat(labels, dim=0).to(device=device, dtype=torch.long)
    return train_x, train_y


def _fit_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    probe = torch.nn.Linear(train_x.shape[-1], num_classes, bias=True).to(device=train_x.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    final_loss = 0.0
    for _ in range(epochs):
        logits = probe(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    with torch.no_grad():
        accuracy = float((probe(train_x).argmax(dim=-1) == train_y).float().mean().item())
        weight = probe.weight.detach().clone()
        bias = probe.bias.detach().clone()
    return weight, bias, accuracy, final_loss


def probe_guided_low_rank_weights(
    probe_weight: torch.Tensor,
    *,
    rank: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probe_weight.ndim != 2:
        raise ValueError("probe_guided_low_rank_weights expects a rank-2 probe weight matrix.")
    _, singular_values, vh = torch.linalg.svd(probe_weight.float(), full_matrices=False)
    effective_rank = max(1, min(rank, int(singular_values.numel()), int(vh.shape[0])))
    basis = vh[:effective_rank].transpose(0, 1).contiguous()
    coeff = singular_values[:effective_rank]
    coeff = coeff / coeff.max().clamp_min(1e-6)
    coeff = coeff * float(scale)
    sqrt_coeff = coeff.clamp_min(0.0).sqrt()
    down_weight = torch.diag(sqrt_coeff) @ basis.transpose(0, 1)
    up_weight = basis @ torch.diag(sqrt_coeff)
    return down_weight, up_weight, coeff


def probe_guided_affine_scale(
    probe_weight: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    if probe_weight.ndim != 2:
        raise ValueError("probe_guided_affine_scale expects a rank-2 probe weight matrix.")
    strength = probe_weight.float().pow(2).sum(dim=0)
    strength = strength / strength.max().clamp_min(1e-6)
    return strength * float(scale)


def _apply_probe_head_or_adapter_warmstart(
    model: PacketRoutingModel,
    *,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    route_mode: str,
    temperature: float,
    estimator: str,
    target_kind: str,
) -> dict[str, Any]:
    warm_cfg = dict(cfg.get("training", {}).get("probe_warmstart", {}) or {})
    if target_kind == "readout_adapter":
        warm_cfg = dict(cfg.get("training", {}).get("probe_adapter_warmstart", {}) or {})
    if not bool(warm_cfg.get("enabled", False)):
        return {}

    split = str(warm_cfg.get("split", "train"))
    num_batches = int(warm_cfg.get("num_batches", 8))
    batch_size = int(warm_cfg.get("batch_size", cfg["training"].get("batch_size", 64)))
    epochs = int(warm_cfg.get("epochs", 200))
    lr = float(warm_cfg.get("lr", 0.05))
    weight_decay = float(warm_cfg.get("weight_decay", 1e-4))
    final_query_only = bool(warm_cfg.get("final_query_only", True))
    feature_source = str(warm_cfg.get("feature_source", "final_readout_input"))
    train_x, train_y = _collect_probe_dataset(
        model,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        cfg=cfg,
        device=device,
        route_mode=route_mode,
        temperature=temperature,
        estimator=estimator,
        split=split,
        num_batches=num_batches,
        batch_size=batch_size,
        final_query_only=final_query_only,
        feature_source=feature_source,
    )
    probe_weight, probe_bias, accuracy, final_loss = _fit_linear_probe(
        train_x,
        train_y,
        num_classes=model.num_classes,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    summary = {
        "probe_warmstart_split": split,
        "probe_warmstart_num_batches": num_batches,
        "probe_warmstart_batch_size": batch_size,
        "probe_warmstart_final_query_only": final_query_only,
        "probe_warmstart_feature_source": feature_source,
        "probe_warmstart_num_examples": int(train_x.shape[0]),
        "probe_warmstart_epochs": epochs,
        "probe_warmstart_lr": lr,
        "probe_warmstart_weight_decay": weight_decay,
        "probe_warmstart_train_accuracy": accuracy,
        "probe_warmstart_final_loss": final_loss,
    }
    if target_kind == "readout_head":
        if not isinstance(model.readout, StandardizedMLPReadoutHead):
            raise ValueError("training.probe_warmstart requires readout_head_mode=mlp.")
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0).clamp_min(1e-5)
        model.readout.reset_parameters()
        model.readout.set_standardizer(mean, std)
        optimizer = torch.optim.AdamW(model.readout.parameters(), lr=lr, weight_decay=weight_decay)
        model.readout.train()
        head_loss = 0.0
        for _ in range(epochs):
            logits = model.readout(train_x)
            loss = F.cross_entropy(logits, train_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            head_loss = float(loss.detach().item())
        model.readout.eval()
        with torch.no_grad():
            head_accuracy = float((model.readout(train_x).argmax(dim=-1) == train_y).float().mean().item())
        summary.update(
            {
                "probe_warmstart_enabled": True,
                "probe_warmstart_train_accuracy": head_accuracy,
                "probe_warmstart_final_loss": head_loss,
            }
        )
        return summary

    adapter_target = str(warm_cfg.get("adapter_target", "readout_adapter"))
    init_scale = float(warm_cfg.get("init_scale", 0.1))
    adapter = getattr(model, adapter_target, None)
    if isinstance(adapter, LowRankAdapter):
        down_weight, up_weight, coeff = probe_guided_low_rank_weights(
            probe_weight,
            rank=int(adapter.down.out_features),
            scale=init_scale,
        )
        adapter.down.weight.data.copy_(down_weight.to(device=adapter.down.weight.device, dtype=adapter.down.weight.dtype))
        adapter.up.weight.data.copy_(up_weight.to(device=adapter.up.weight.device, dtype=adapter.up.weight.dtype))
        summary.update(
            {
                "probe_adapter_warmstart_enabled": True,
                "probe_adapter_warmstart_target": adapter_target,
                "probe_adapter_warmstart_adapter_mode": "low_rank",
                "probe_adapter_warmstart_init_scale": init_scale,
                "probe_adapter_warmstart_effective_rank": int(coeff.numel()),
                "probe_adapter_warmstart_coeff_max": float(coeff.max().item()),
            }
        )
        return summary
    if isinstance(adapter, AffineAdapter):
        scale_vec = probe_guided_affine_scale(probe_weight, scale=init_scale)
        adapter.scale.data.copy_(scale_vec.to(device=adapter.scale.device, dtype=adapter.scale.dtype))
        adapter.bias.data.zero_()
        summary.update(
            {
                "probe_adapter_warmstart_enabled": True,
                "probe_adapter_warmstart_target": adapter_target,
                "probe_adapter_warmstart_adapter_mode": "affine",
                "probe_adapter_warmstart_init_scale": init_scale,
                "probe_adapter_warmstart_scale_mean": float(scale_vec.mean().item()),
                "probe_adapter_warmstart_scale_max": float(scale_vec.max().item()),
            }
        )
        return summary
    raise ValueError(
        f"training.probe_adapter_warmstart requires {adapter_target} to be a LowRankAdapter or AffineAdapter."
    )


def apply_probe_adapter_warmstart(
    model: PacketRoutingModel,
    *,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    route_mode: str,
    temperature: float,
    estimator: str,
) -> dict[str, Any]:
    return _apply_probe_head_or_adapter_warmstart(
        model,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        cfg=cfg,
        device=device,
        route_mode=route_mode,
        temperature=temperature,
        estimator=estimator,
        target_kind="readout_adapter",
    )


def build_supervised_optimizer(
    model: PacketRoutingModel,
    train_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    controller_lr_scale = float(train_cfg.get("controller_lr_scale", 1.0))
    controller_weight_decay = float(train_cfg.get("controller_weight_decay", weight_decay))
    controller_prefixes = tuple(
        train_cfg.get(
            "controller_prefixes",
            [
                "control_",
                "wait_",
                "core.router_mlp",
                "core.router_out",
                "core.router_act_out",
                "core.router_wait_out",
                "release_",
            ],
        )
    )

    controller_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in controller_prefixes):
            controller_params.append(parameter)
        else:
            other_params.append(parameter)

    if controller_lr_scale == 1.0 and controller_weight_decay == weight_decay:
        param_groups = [{"params": other_params + controller_params, "lr": lr, "weight_decay": weight_decay}]
    else:
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
        if controller_params:
            param_groups.append(
                {
                    "params": controller_params,
                    "lr": lr * controller_lr_scale,
                    "weight_decay": controller_weight_decay,
                }
            )

    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups)
    return torch.optim.AdamW(param_groups)


def load_checkpoint(
    path: str | Path,
    model: PacketRoutingModel,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload["model"]
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        current_state = model.state_dict()
        filtered_state = {
            name: tensor
            for name, tensor in state_dict.items()
            if name in current_state and current_state[name].shape == tensor.shape
        }
        model.load_state_dict(filtered_state, strict=False)
    if optimizer is not None and "optimizer" in payload:
        if strict:
            optimizer.load_state_dict(payload["optimizer"])
        else:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except ValueError:
                pass
    return payload


def resolve_resume_checkpoint(cfg: dict[str, Any]) -> tuple[str | Path | None, bool]:
    resume_cfg = cfg.get("resume")
    if not resume_cfg:
        return None, bool(cfg.get("resume_strict", True))
    if isinstance(resume_cfg, dict):
        checkpoint = resume_cfg.get("checkpoint")
        if checkpoint is None:
            raise ValueError("resume.checkpoint must be set when resume is a mapping.")
        strict = bool(resume_cfg.get("strict", cfg.get("resume_strict", True)))
        return checkpoint, strict
    return resume_cfg, bool(cfg.get("resume_strict", True))


def resolve_checkpoint_from_run_dir(run_dir: Path, explicit: str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates = [
        run_dir / "hybrid_es_best.pt",
        run_dir / "hard_st_best.pt",
        run_dir / "reinforce_best.pt",
        run_dir / "soft_best.pt",
    ]
    candidates.extend(sorted(run_dir.glob("*_best.pt")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No *_best checkpoint found in {run_dir}.")


def benchmark_model_config(base_model_cfg: dict[str, Any], benchmark) -> dict[str, Any]:
    model_cfg = {
        **base_model_cfg,
        "num_nodes": benchmark.num_nodes,
        "obs_dim": benchmark.obs_dim,
        "num_classes": benchmark.num_classes,
        "max_total_steps": benchmark.config.get(
            "max_total_steps",
            benchmark.config.get("seq_len", 2) * max(benchmark.num_nodes, 1) * 2,
        ),
    }
    if hasattr(benchmark, "query_offset"):
        model_cfg["query_offset"] = int(benchmark.query_offset)
    if hasattr(benchmark, "query_cardinality"):
        model_cfg["query_cardinality"] = int(benchmark.query_cardinality)
    if hasattr(benchmark, "payload_cardinality"):
        model_cfg["payload_cardinality"] = int(benchmark.payload_cardinality)
    return model_cfg


def load_teacher_distillation(
    cfg: dict[str, Any],
    *,
    benchmark,
    device: torch.device,
    section: str = "teacher",
) -> TeacherDistillation | None:
    teacher_cfg = dict(cfg.get(section, {}) or {})
    weights = {
        "logits_weight": float(teacher_cfg.get("logits_weight", 0.0)),
        "route_weight": float(teacher_cfg.get("route_weight", 0.0)),
        "route_action_weight": float(teacher_cfg.get("route_action_weight", 0.0)),
        "control_prob_weight": float(teacher_cfg.get("control_prob_weight", 0.0)),
        "release_prob_weight": float(teacher_cfg.get("release_prob_weight", 0.0)),
        "wait_prob_weight": float(teacher_cfg.get("wait_prob_weight", 0.0)),
        "control_state_weight": float(teacher_cfg.get("control_state_weight", 0.0)),
        "memory_read_weight": float(teacher_cfg.get("memory_read_weight", 0.0)),
        "factorized_content_weight": float(teacher_cfg.get("factorized_content_weight", 0.0)),
        "factorized_query_weight": float(teacher_cfg.get("factorized_query_weight", 0.0)),
    }
    if max(weights.values(), default=0.0) <= 0.0:
        return None

    run_dir = teacher_cfg.get("run_dir")
    if not run_dir:
        raise ValueError(f"{section} distillation requires {section}.run_dir when any teacher weight is positive.")
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text())
    config_path = Path(teacher_cfg.get("config_path") or summary_payload.get("config_path") or (run_dir / "config.yaml"))
    if not config_path.exists():
        if summary_path.exists():
            raise FileNotFoundError(f"Teacher config not found: {config_path}")
        raise FileNotFoundError(
            f"Teacher run requires either {summary_path} or {run_dir / 'config.yaml'}"
        )
    teacher_base_cfg = load_config(str(config_path))
    teacher_model = PacketRoutingModel(
        benchmark_model_config(teacher_base_cfg["model"], benchmark)
    ).to(device)
    checkpoint_path = resolve_checkpoint_from_run_dir(run_dir, teacher_cfg.get("checkpoint"))
    load_checkpoint(
        checkpoint_path,
        teacher_model,
        strict=bool(teacher_cfg.get("strict", teacher_base_cfg.get("resume_strict", True))),
    )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    return TeacherDistillation(
        model=teacher_model,
        route_mode=str(teacher_cfg.get("route_mode", "hard")),
        temperature=float(teacher_cfg.get("temperature", 1.0)),
        estimator=str(teacher_cfg.get("estimator", "straight_through")),
        distill_temperature=float(teacher_cfg.get("distill_temperature", 1.0)),
        target_scope=str(teacher_cfg.get("target_scope", "all")),
        logits_weight=weights["logits_weight"],
        route_weight=weights["route_weight"],
        route_action_weight=weights["route_action_weight"],
        control_prob_weight=weights["control_prob_weight"],
        release_prob_weight=weights["release_prob_weight"],
        wait_prob_weight=weights["wait_prob_weight"],
        control_state_weight=weights["control_state_weight"],
        memory_read_weight=weights["memory_read_weight"],
        factorized_content_weight=weights["factorized_content_weight"],
        factorized_query_weight=weights["factorized_query_weight"],
        start_step=int(teacher_cfg.get("start_step", 0)),
        stop_step=int(teacher_cfg.get("stop_step", -1)),
        scale_start=float(teacher_cfg.get("scale_start", 1.0)),
        scale_end=float(teacher_cfg.get("scale_end", 1.0)),
        scale_schedule_steps=int(teacher_cfg.get("scale_schedule_steps", 0)),
        dropout_prob_start=float(teacher_cfg.get("dropout_prob_start", teacher_cfg.get("dropout_prob", 0.0))),
        dropout_prob_end=float(teacher_cfg.get("dropout_prob_end", teacher_cfg.get("dropout_prob", 0.0))),
        dropout_schedule_steps=int(teacher_cfg.get("dropout_prob_schedule_steps", 0)),
    )


def load_parameter_anchor(
    cfg: dict[str, Any],
    *,
    model: PacketRoutingModel,
    device: torch.device,
) -> ParameterAnchor | None:
    anchor_cfg = dict(cfg.get("parameter_anchor") or {})
    base_weight = float(anchor_cfg.get("weight", 0.0))
    weight_start = float(anchor_cfg.get("weight_start", base_weight))
    weight_end = float(anchor_cfg.get("weight_end", base_weight))
    if max(weight_start, weight_end) <= 0.0:
        return None

    run_dir_value = anchor_cfg.get("run_dir")
    checkpoint_value = anchor_cfg.get("checkpoint")
    if run_dir_value:
        checkpoint_path = resolve_checkpoint_from_run_dir(Path(run_dir_value), checkpoint_value)
    elif checkpoint_value:
        checkpoint_path = Path(str(checkpoint_value))
    else:
        raise ValueError("parameter_anchor requires either parameter_anchor.run_dir or parameter_anchor.checkpoint.")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model"]
    include_prefixes = tuple(str(prefix) for prefix in anchor_cfg.get("include_prefixes", []))
    exclude_prefixes = tuple(str(prefix) for prefix in anchor_cfg.get("exclude_prefixes", []))
    current_state = model.state_dict()
    anchor_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name not in current_state or current_state[name].shape != tensor.shape:
            continue
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if not torch.is_floating_point(current_state[name]):
            continue
        anchor_tensors[name] = tensor.to(device=device, dtype=current_state[name].dtype)

    if not anchor_tensors:
        raise ValueError(f"parameter_anchor copied zero parameters from {checkpoint_path}.")

    return ParameterAnchor(
        tensors=anchor_tensors,
        weight_start=weight_start,
        weight_end=weight_end,
        weight_schedule_steps=int(anchor_cfg.get("weight_schedule_steps", 0)),
        start_step=int(anchor_cfg.get("start_step", 0)),
        stop_step=int(anchor_cfg.get("stop_step", -1)),
        p=float(anchor_cfg.get("p", 2.0)),
        normalize=bool(anchor_cfg.get("normalize", True)),
    )


def parameter_anchor_weight(anchor: ParameterAnchor, step: int) -> float:
    if step < anchor.start_step:
        return 0.0
    if anchor.stop_step >= 0 and step >= anchor.stop_step:
        return 0.0
    if anchor.weight_schedule_steps > 0:
        frac = min(1.0, max(0.0, (step - anchor.start_step) / max(1, anchor.weight_schedule_steps)))
        return float(anchor.weight_start + frac * (anchor.weight_end - anchor.weight_start))
    return float(anchor.weight_end)


def compute_parameter_anchor_loss(
    *,
    model: PacketRoutingModel,
    anchor: ParameterAnchor,
    step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = parameter_anchor_weight(anchor, step)
    if weight <= 0.0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "parameter_anchor_loss": 0.0,
            "parameter_anchor_loss_raw": 0.0,
            "parameter_anchor_weight": 0.0,
            "parameter_anchor_param_count": 0.0,
        }

    raw_total = torch.zeros((), device=device, dtype=torch.float32)
    param_count = 0
    anchored_params = 0
    p_value = max(1.0, float(anchor.p))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        reference = anchor.tensors.get(name)
        if reference is None:
            continue
        diff = (parameter.float() - reference.float()).abs()
        if p_value == 1.0:
            raw_total = raw_total + diff.sum()
        else:
            raw_total = raw_total + diff.pow(p_value).sum()
        param_count += diff.numel()
        anchored_params += 1
    if anchored_params == 0:
        raise ValueError("parameter_anchor did not match any trainable parameters.")
    if anchor.normalize and param_count > 0:
        raw_total = raw_total / float(param_count)
    total = raw_total.to(dtype=dtype) * float(weight)
    return total, {
        "parameter_anchor_loss": float(total.detach().item()),
        "parameter_anchor_loss_raw": float(raw_total.detach().item()),
        "parameter_anchor_weight": float(weight),
        "parameter_anchor_param_count": float(param_count),
        "parameter_anchor_matched_tensors": float(anchored_params),
    }


def teacher_step_controls(teacher: TeacherDistillation, step: int) -> tuple[float, float]:
    if step < teacher.start_step:
        return 0.0, 0.0
    if teacher.stop_step >= 0 and step >= teacher.stop_step:
        return 0.0, 0.0

    if teacher.scale_schedule_steps > 0:
        frac = min(1.0, max(0.0, (step - teacher.start_step) / max(1, teacher.scale_schedule_steps)))
        scale = teacher.scale_start + frac * (teacher.scale_end - teacher.scale_start)
    else:
        scale = teacher.scale_end

    if teacher.dropout_schedule_steps > 0:
        frac = min(1.0, max(0.0, (step - teacher.start_step) / max(1, teacher.dropout_schedule_steps)))
        dropout_prob = teacher.dropout_prob_start + frac * (
            teacher.dropout_prob_end - teacher.dropout_prob_start
        )
    else:
        dropout_prob = teacher.dropout_prob_end
    return float(scale), float(dropout_prob)


def teacher_sample_weights(
    batch: BenchmarkBatch,
    target_scope: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.ones(batch.labels.shape[0], device=device, dtype=dtype)
    if target_scope == "all":
        return weights
    if target_scope == "final_query_only":
        mask = batch.metadata.get("needs_final_query")
        if mask is None:
            return torch.zeros_like(weights)
        return mask.to(device=device, dtype=dtype)
    if target_scope == "delayed_only":
        delayed = (
            (batch.modes == LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT)
            | (batch.modes == LONG_MEMORY_MODE_DELAY_TO_FINAL_QUERY)
        )
        return delayed.to(device=device, dtype=dtype)
    raise ValueError(f"Unknown teacher.target_scope: {target_scope}")


def _weighted_kl_from_probs(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-6
    teacher_probs = teacher_probs.float().clamp_min(eps)
    student_probs = student_probs.float().clamp_min(eps)
    kl = (teacher_probs * (teacher_probs.log() - student_probs.log())).sum(dim=-1)
    while sample_weights.dim() < kl.dim():
        sample_weights = sample_weights.unsqueeze(-1)
    return (kl * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def _weighted_mse(
    student_value: torch.Tensor,
    teacher_value: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    diff = (student_value.float() - teacher_value.float()).pow(2)
    while diff.dim() > sample_weights.dim():
        sample_weights = sample_weights.unsqueeze(-1)
    reduce_dims = tuple(range(sample_weights.dim(), diff.dim()))
    if reduce_dims:
        diff = diff.mean(dim=reduce_dims)
    return (diff * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def _weighted_action_ce_from_probs(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-6
    student_log_probs = student_probs.float().clamp_min(eps).log()
    teacher_actions = teacher_probs.float().argmax(dim=-1)
    nll = -student_log_probs.gather(-1, teacher_actions.unsqueeze(-1)).squeeze(-1)
    while sample_weights.dim() < nll.dim():
        sample_weights = sample_weights.unsqueeze(-1)
    return (nll * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def compute_teacher_distillation_loss(
    *,
    batch: BenchmarkBatch,
    student_output,
    teacher_output,
    teacher: TeacherDistillation,
    device: torch.device,
    scale: float = 1.0,
    dropout_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = student_output.logits.dtype
    sample_weights = teacher_sample_weights(batch, teacher.target_scope, device=device, dtype=dtype)
    if dropout_prob > 0.0:
        keep_mask = (torch.rand_like(sample_weights) >= dropout_prob).to(sample_weights.dtype)
        sample_weights = sample_weights * keep_mask
    if float(sample_weights.sum().detach().item()) <= 0.0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "teacher_distill_loss": 0.0,
            "teacher_distill_loss_raw": 0.0,
            "teacher_scale": float(scale),
            "teacher_dropout_prob": float(dropout_prob),
            "teacher_effective_weight_mean": 0.0,
        }

    raw_total_loss = torch.zeros((), device=device, dtype=dtype)
    metrics: dict[str, float] = {}
    distill_temperature = max(1e-3, teacher.distill_temperature)

    if teacher.logits_weight > 0.0:
        student_log_probs = F.log_softmax(student_output.logits.float() / distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_output.logits.float() / distill_temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        logits_loss = ((kl * sample_weights.float()).sum() / sample_weights.float().sum().clamp_min(1.0)) * (
            distill_temperature**2
        )
        raw_total_loss = raw_total_loss + teacher.logits_weight * logits_loss.to(dtype)
        metrics["teacher_logits_loss"] = float(logits_loss.detach().item())

    student_trace = student_output.trace or {}
    teacher_trace = teacher_output.trace or {}
    if (
        teacher.route_action_weight > 0.0
        and "router_probs" in student_trace
        and "router_probs" in teacher_trace
        and student_trace["router_probs"].shape == teacher_trace["router_probs"].shape
    ):
        route_action_loss = _weighted_action_ce_from_probs(
            student_trace["router_probs"],
            teacher_trace["router_probs"],
            sample_weights,
        )
        raw_total_loss = raw_total_loss + teacher.route_action_weight * route_action_loss.to(dtype)
        metrics["teacher_route_action_loss"] = float(route_action_loss.detach().item())
    trace_specs = [
        ("router_probs", teacher.route_weight, "teacher_route_loss", _weighted_kl_from_probs),
        ("control_prob", teacher.control_prob_weight, "teacher_control_prob_loss", _weighted_mse),
        ("release_prob", teacher.release_prob_weight, "teacher_release_prob_loss", _weighted_mse),
        ("wait_prob", teacher.wait_prob_weight, "teacher_wait_prob_loss", _weighted_mse),
        ("control_state", teacher.control_state_weight, "teacher_control_state_loss", _weighted_mse),
        ("memory_read_state", teacher.memory_read_weight, "teacher_memory_read_loss", _weighted_mse),
        (
            "factorized_content_hidden",
            teacher.factorized_content_weight,
            "teacher_factorized_content_loss",
            _weighted_mse,
        ),
        (
            "factorized_query_hidden",
            teacher.factorized_query_weight,
            "teacher_factorized_query_loss",
            _weighted_mse,
        ),
    ]
    for key, weight, metric_name, loss_fn in trace_specs:
        if weight <= 0.0:
            continue
        if key not in student_trace or key not in teacher_trace:
            continue
        if student_trace[key].shape != teacher_trace[key].shape:
            continue
        trace_loss = loss_fn(student_trace[key], teacher_trace[key], sample_weights)
        raw_total_loss = raw_total_loss + weight * trace_loss.to(dtype)
        metrics[metric_name] = float(trace_loss.detach().item())

    total_loss = raw_total_loss * float(scale)
    metrics["teacher_distill_loss_raw"] = float(raw_total_loss.detach().item())
    metrics["teacher_distill_loss"] = float(total_loss.detach().item())
    metrics["teacher_scale"] = float(scale)
    metrics["teacher_dropout_prob"] = float(dropout_prob)
    metrics["teacher_effective_weight_mean"] = float(sample_weights.detach().float().mean().item())
    return total_loss, metrics


def reinforce_advantages(
    reward: torch.Tensor,
    method_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    reward = reward.float()
    baseline_mode = str(method_cfg.get("baseline_mode", "batch_mean"))
    if baseline_mode == "none":
        baseline = torch.zeros((), device=reward.device, dtype=reward.dtype)
        centered = reward
    else:
        baseline = reward.mean()
        centered = reward - baseline
    advantage_mode = str(method_cfg.get("advantage_mode", "standardize"))
    if advantage_mode == "standardize":
        denom = centered.std(unbiased=False).clamp_min(1e-6)
        return centered / denom, baseline
    if advantage_mode == "center":
        return centered, baseline
    if advantage_mode == "none":
        return reward, baseline
    raise ValueError(f"Unknown advantage_mode: {advantage_mode}")


def run_supervised_phase(
    *,
    phase_name: str,
    model: PacketRoutingModel,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    results_dir: Path,
    logger: JsonlLogger,
    route_mode: str,
    temperature: float,
    estimator: str,
    steps_override: int | None = None,
) -> dict[str, Any]:
    train_cfg = cfg["training"]
    system_cfg = cfg.get("system", {})
    objective_cfg = cfg["objective"]
    schedule_cfg = cfg.get("objective_schedule", {})
    routing_cfg = cfg.get("routing", {})
    amp_enabled = bool(system_cfg.get("amp", False))
    amp_dtype = str(system_cfg.get("amp_dtype", "bf16"))
    teacher = load_teacher_distillation(cfg, benchmark=benchmark, device=device)
    proxy_teacher = load_teacher_distillation(cfg, benchmark=benchmark, device=device, section="proxy_teacher")
    parameter_anchor = load_parameter_anchor(cfg, model=model, device=device)
    auxiliary_train_benchmarks = load_auxiliary_train_benchmarks(cfg)
    auxiliary_eval_benchmarks = load_auxiliary_eval_benchmarks(cfg, section="training")
    partial_init_summary = apply_partial_init(model, cfg.get("partial_init")) if not cfg.get("resume") else {}
    resume_path, resume_strict = resolve_resume_checkpoint(cfg)
    probe_warmstart_summary = (
        apply_probe_warmstart(
            model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg=cfg,
            device=device,
            route_mode=route_mode,
            temperature=temperature,
            estimator=estimator,
        )
        if not cfg.get("resume")
        else {}
    )
    probe_adapter_warmstart_summary = (
        apply_probe_adapter_warmstart(
            model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg=cfg,
            device=device,
            route_mode=route_mode,
            temperature=temperature,
            estimator=estimator,
        )
        if not cfg.get("resume")
        else {}
    )

    trainable_summary = configure_trainable_parameters(model, train_cfg)
    optimizer = build_supervised_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    start_step = 0
    best_val = -1.0
    best_path = results_dir / f"{phase_name}_best.pt"
    last_path = results_dir / f"{phase_name}_last.pt"
    stability_guard_cfg = train_cfg.get("stability_guard", {}) if isinstance(train_cfg.get("stability_guard"), dict) else {}
    stability_guard_restore_path: Path | None = None
    stability_guard_restore_label = "best"
    if stability_guard_cfg.get("restore_checkpoint") is not None:
        stability_guard_restore_path = Path(str(stability_guard_cfg["restore_checkpoint"]))
        stability_guard_restore_label = "checkpoint"
    elif stability_guard_cfg.get("restore_run_dir") is not None:
        stability_guard_restore_path = resolve_checkpoint_from_run_dir(Path(str(stability_guard_cfg["restore_run_dir"])))
        stability_guard_restore_label = "run_dir"

    if resume_path:
        payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            strict=resume_strict,
        )
        start_step = int(payload.get("step", 0))

    configured_total_steps = int(steps_override or train_cfg["train_steps"])
    if "train_steps_delta" in train_cfg:
        total_steps = start_step + int(train_cfg["train_steps_delta"])
    else:
        total_steps = configured_total_steps
    eval_compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, total_steps)
    model.train()
    phase_start_time = time.time()
    peak_train_memory_mb = 0.0
    best_val_metrics: dict[str, Any] | None = None
    best_val_score = initial_selection_score(train_cfg)
    best_metric_name = str(train_cfg.get("selection_metric", "accuracy"))
    stability_guard_consecutive_violations = 0
    stability_guard_rollbacks = 0
    stability_guard_cooldown = 0
    should_stop_early = False
    progress = tqdm(range(start_step, total_steps), disable=False)
    for step in progress:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch = benchmark.sample_batch(
            batch_size=int(train_cfg["batch_size"]),
            split="train",
            step=step,
            device=device,
        )
        train_temperature = current_training_temperature(cfg["method"], step)
        compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, step)
        step_routing_cfg = current_routing_cfg(routing_cfg, step)
        forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        memory_payload_targets, memory_payload_mask, memory_payload_weight = build_memory_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        control_targets, control_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        wait_targets, wait_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        release_targets, release_mask, release_weight, release_positive_weight = build_release_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        task_sample_weights = build_task_sample_weights(
            batch,
            train_cfg,
            device=batch.labels.device,
            dtype=batch.observations.dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        return_trace = teacher is not None
        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(
                observations=batch.observations,
                labels=batch.labels,
                route_mode=route_mode,
                compute_penalties=compute_penalties,
                temperature=train_temperature,
                estimator=estimator,
                truncate_bptt_steps=int(train_cfg.get("truncate_bptt_steps", 0)),
                detach_prefix_steps=int(train_cfg.get("detach_prefix_steps", 0)),
                late_window_steps=int(train_cfg.get("late_window_steps", 0)),
                forced_actions=forced_actions,
                action_masks=action_masks,
                oracle_actions=oracle_actions,
                oracle_action_mask=oracle_action_mask,
                oracle_route_weight=oracle_route_weight,
                delay_write_targets=batch.delay_write_targets,
                delay_write_mask=batch.delay_write_mask,
                delay_write_weight=delay_write_weight,
                memory_payload_targets=memory_payload_targets,
                memory_payload_mask=memory_payload_mask,
                memory_payload_weight=memory_payload_weight,
                control_targets=control_targets,
                control_mask=control_mask,
                control_weight=control_weight,
                anti_exit_mask=anti_exit_mask,
                anti_exit_weight=anti_exit_weight,
                wait_targets=wait_targets,
                wait_mask=wait_mask,
                wait_weight=wait_weight,
                wait_positive_weight=wait_positive_weight,
                wait_negative_weight=wait_negative_weight,
                release_targets=release_targets,
                release_mask=release_mask,
                release_weight=release_weight,
                release_positive_weight=release_positive_weight,
                factorized_payload_targets=batch.metadata.get("payload"),
                factorized_query_targets=batch.metadata.get("query"),
                task_sample_weights=task_sample_weights,
                final_query_mask=batch.metadata.get("needs_final_query"),
                return_trace=return_trace,
            )
            teacher_loss = torch.zeros((), device=device, dtype=output.loss.dtype)
            teacher_metrics: dict[str, float] = {}
            if teacher is not None:
                teacher_scale, teacher_dropout_prob = teacher_step_controls(teacher, step)
                with torch.no_grad():
                    teacher_output = teacher.model(
                        observations=batch.observations,
                        labels=batch.labels,
                        route_mode=teacher.route_mode,
                        compute_penalties=compute_penalties,
                        temperature=teacher.temperature,
                        estimator=teacher.estimator,
                        truncate_bptt_steps=0,
                        final_query_mask=batch.metadata.get("needs_final_query"),
                        return_trace=True,
                    )
                teacher_loss, teacher_metrics = compute_teacher_distillation_loss(
                    batch=batch,
                    student_output=output,
                    teacher_output=teacher_output,
                    teacher=teacher,
                    device=device,
                    scale=teacher_scale,
                    dropout_prob=teacher_dropout_prob,
                )
            auxiliary_loss = torch.zeros((), device=device, dtype=output.loss.dtype)
            auxiliary_metrics: dict[str, float] = {}
            for aux_source in auxiliary_train_benchmarks:
                if aux_source.loss_weight <= 0.0 and aux_source.agreement_weight <= 0.0:
                    continue
                aux_batch = aux_source.benchmark.sample_batch(
                    batch_size=aux_source.batch_size,
                    split=aux_source.split,
                    step=step,
                    device=device,
                )
                aux_forced_actions, aux_action_masks, aux_oracle_actions, aux_oracle_action_mask, aux_oracle_route_weight, aux_delay_write_weight = build_routing_controls(
                    aux_batch,
                    step_routing_cfg,
                    split="train",
                )
                aux_memory_payload_targets, aux_memory_payload_mask, aux_memory_payload_weight = build_memory_controls(
                    aux_batch,
                    step_routing_cfg,
                    split="train",
                )
                aux_control_targets, aux_control_mask, aux_control_weight, aux_anti_exit_mask, aux_anti_exit_weight = build_control_controls(
                    aux_batch,
                    step_routing_cfg,
                    split="train",
                )
                aux_wait_targets, aux_wait_mask, aux_wait_weight, aux_wait_positive_weight, aux_wait_negative_weight = build_wait_controls(
                    aux_batch,
                    step_routing_cfg,
                    split="train",
                )
                aux_release_targets, aux_release_mask, aux_release_weight, aux_release_positive_weight = build_release_controls(
                    aux_batch,
                    step_routing_cfg,
                    split="train",
                )
                aux_task_sample_weights = build_task_sample_weights(
                    aux_batch,
                    aux_source.task_weight_cfg,
                    device=aux_batch.labels.device,
                    dtype=aux_batch.observations.dtype,
                )
                aux_return_trace = proxy_teacher is not None and aux_source.agreement_weight > 0.0
                aux_output = model(
                    observations=aux_batch.observations,
                    labels=aux_batch.labels,
                    route_mode=route_mode,
                    compute_penalties=compute_penalties,
                    temperature=train_temperature,
                    estimator=estimator,
                    truncate_bptt_steps=int(train_cfg.get("truncate_bptt_steps", 0)),
                    detach_prefix_steps=int(train_cfg.get("detach_prefix_steps", 0)),
                    late_window_steps=int(train_cfg.get("late_window_steps", 0)),
                    forced_actions=aux_forced_actions,
                    action_masks=aux_action_masks,
                    oracle_actions=aux_oracle_actions,
                    oracle_action_mask=aux_oracle_action_mask,
                    oracle_route_weight=aux_oracle_route_weight,
                    delay_write_targets=aux_batch.delay_write_targets,
                    delay_write_mask=aux_batch.delay_write_mask,
                    delay_write_weight=aux_delay_write_weight,
                    memory_payload_targets=aux_memory_payload_targets,
                    memory_payload_mask=aux_memory_payload_mask,
                    memory_payload_weight=aux_memory_payload_weight,
                    control_targets=aux_control_targets,
                    control_mask=aux_control_mask,
                    control_weight=aux_control_weight,
                    anti_exit_mask=aux_anti_exit_mask,
                    anti_exit_weight=aux_anti_exit_weight,
                    wait_targets=aux_wait_targets,
                    wait_mask=aux_wait_mask,
                    wait_weight=aux_wait_weight,
                    wait_positive_weight=aux_wait_positive_weight,
                    wait_negative_weight=aux_wait_negative_weight,
                    release_targets=aux_release_targets,
                    release_mask=aux_release_mask,
                    release_weight=aux_release_weight,
                    release_positive_weight=aux_release_positive_weight,
                    task_sample_weights=aux_task_sample_weights,
                    final_query_mask=aux_batch.metadata.get("needs_final_query"),
                    return_trace=aux_return_trace,
                )
                aux_prefix = f"aux_{aux_source.name}"
                if aux_source.loss_weight > 0.0:
                    auxiliary_loss = auxiliary_loss + (aux_output.loss * aux_source.loss_weight)
                auxiliary_metrics[f"{aux_prefix}_loss"] = float(aux_output.loss.detach().item())
                auxiliary_metrics[f"{aux_prefix}_loss_weight"] = float(aux_source.loss_weight)
                auxiliary_metrics[f"{aux_prefix}_agreement_weight"] = float(aux_source.agreement_weight)
                auxiliary_metrics[f"{aux_prefix}_accuracy"] = float(aux_output.stats["accuracy"].detach().float().mean().item())
                final_query_mask = aux_batch.metadata.get("needs_final_query")
                if final_query_mask is not None:
                    final_query_mask = final_query_mask.to(device=device, dtype=torch.bool)
                    if bool(final_query_mask.any().item()):
                        predictions = aux_output.logits.argmax(dim=-1)
                        auxiliary_metrics[f"{aux_prefix}_final_query_accuracy"] = float(
                            (predictions[final_query_mask] == aux_batch.labels[final_query_mask]).float().mean().item()
                        )
                if proxy_teacher is not None and aux_source.agreement_weight > 0.0:
                    with torch.no_grad():
                        aux_teacher_output = proxy_teacher.model(
                            observations=aux_batch.observations,
                            labels=aux_batch.labels,
                            route_mode=proxy_teacher.route_mode,
                            compute_penalties=compute_penalties,
                            temperature=proxy_teacher.temperature,
                            estimator=proxy_teacher.estimator,
                            truncate_bptt_steps=0,
                            final_query_mask=aux_batch.metadata.get("needs_final_query"),
                            return_trace=True,
                        )
                    agreement_loss, agreement_metrics = compute_teacher_distillation_loss(
                        batch=aux_batch,
                        student_output=aux_output,
                        teacher_output=aux_teacher_output,
                        teacher=proxy_teacher,
                        device=device,
                        scale=aux_source.agreement_weight,
                        dropout_prob=0.0,
                    )
                    auxiliary_loss = auxiliary_loss + agreement_loss
                    for key, value in agreement_metrics.items():
                        auxiliary_metrics[f"{aux_prefix}_{key}"] = value
            parameter_anchor_loss = torch.zeros((), device=device, dtype=output.loss.dtype)
            parameter_anchor_metrics: dict[str, float] = {}
            if parameter_anchor is not None:
                parameter_anchor_loss, parameter_anchor_metrics = compute_parameter_anchor_loss(
                    model=model,
                    anchor=parameter_anchor,
                    step=step,
                    device=device,
                    dtype=output.loss.dtype,
                )
            total_loss = output.loss + teacher_loss + auxiliary_loss + parameter_anchor_loss
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(train_cfg.get("grad_clip", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()

        batch_metrics = tensor_dict_mean([summarize_batch(batch, output, benchmark_name)])
        batch_metrics["loss"] = float(total_loss.detach().item())
        batch_metrics["model_loss"] = float(output.loss.detach().item())
        if teacher is not None:
            batch_metrics.update(teacher_metrics)
        if auxiliary_train_benchmarks:
            batch_metrics["auxiliary_loss"] = float(auxiliary_loss.detach().item())
            batch_metrics.update(auxiliary_metrics)
        if parameter_anchor is not None:
            batch_metrics.update(parameter_anchor_metrics)
        batch_metrics.update(
            {
                "phase": phase_name,
                "split": "train",
                "step": step,
                "examples_per_sec": float(train_cfg["batch_size"]) / max(1e-6, time.time() - start_time),
                "peak_memory_mb": (
                    float(torch.cuda.max_memory_allocated(device) / (1024**2))
                    if device.type == "cuda"
                    else 0.0
                ),
                "temperature": train_temperature,
            }
        )
        peak_train_memory_mb = max(peak_train_memory_mb, float(batch_metrics["peak_memory_mb"]))
        logger.write(batch_metrics)
        progress.set_description(
            f"{phase_name} step={step} acc={batch_metrics['accuracy']:.3f} loss={batch_metrics['loss']:.3f}"
        )

        val_every = int(train_cfg.get("val_every", 25))
        if (step + 1) % val_every == 0 or step + 1 == total_steps:
            val_metrics = evaluate_model(
                model=model,
                benchmark=benchmark,
                device=device,
                benchmark_name=benchmark_name,
                split="val",
                num_batches=int(train_cfg.get("val_batches", 8)),
                batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
                route_mode=route_mode,
                compute_penalties=eval_compute_penalties,
                temperature=temperature,
                estimator=estimator,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                routing_cfg=None,
            )
            selection_eval_metrics: dict[str, Any] = {}
            for eval_source in auxiliary_eval_benchmarks:
                eval_metrics = evaluate_model(
                    model=model,
                    benchmark=eval_source.benchmark,
                    device=device,
                    benchmark_name=eval_source.benchmark_name,
                    split=eval_source.split,
                    num_batches=eval_source.num_batches,
                    batch_size=eval_source.batch_size,
                    route_mode=route_mode,
                    compute_penalties=eval_compute_penalties,
                    temperature=temperature,
                    estimator=estimator,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    routing_cfg=None,
                )
                eval_metrics.update({"phase": phase_name, "split": f"val_proxy_{eval_source.name}", "step": step})
                peak_train_memory_mb = max(peak_train_memory_mb, float(eval_metrics["peak_memory_mb"]))
                logger.write(eval_metrics)
                selection_eval_metrics[eval_source.name] = eval_metrics
            if selection_eval_metrics:
                val_metrics["selection_eval"] = selection_eval_metrics
            val_metrics.update({"phase": phase_name, "split": "val", "step": step})
            stability_guard_metrics = evaluate_stability_guard(train_cfg, val_metrics, step=step)
            if stability_guard_metrics is not None:
                val_metrics["stability_guard"] = {
                    **stability_guard_metrics,
                    "consecutive_violations_before_update": float(stability_guard_consecutive_violations),
                    "rollbacks_before_update": float(stability_guard_rollbacks),
                    "cooldown_before_update": float(stability_guard_cooldown),
                }
            peak_train_memory_mb = max(peak_train_memory_mb, float(val_metrics["peak_memory_mb"]))
            logger.write(val_metrics)
            val_score, metric_name = validation_score(val_metrics, cfg, section="training")
            guard_passed = True
            if stability_guard_metrics is not None:
                if stability_guard_cooldown > 0:
                    stability_guard_cooldown -= 1
                elif not bool(stability_guard_metrics["passed"]):
                    guard_passed = False
                    stability_guard_consecutive_violations += 1
                    violation_limit = max(1, int(stability_guard_metrics["max_consecutive_violations"]))
                    if stability_guard_consecutive_violations >= violation_limit:
                        restore_path = best_path if best_path.exists() else stability_guard_restore_path
                        if restore_path is not None and restore_path.exists():
                            load_checkpoint(
                                restore_path,
                                model,
                                optimizer,
                                strict=(restore_path == best_path),
                            )
                            stability_guard_rollbacks += 1
                            stability_guard_consecutive_violations = 0
                            stability_guard_cooldown = max(0, int(stability_guard_metrics["cooldown_evals"]))
                            logger.write(
                                {
                                    "phase": phase_name,
                                    "split": "stability_guard",
                                    "step": step,
                                    "guard_action": "rollback",
                                    "restore_source": stability_guard_restore_label if restore_path != best_path else "best",
                                    "restore_path": str(restore_path),
                                    "rollbacks": float(stability_guard_rollbacks),
                                    "cooldown_evals": float(stability_guard_cooldown),
                                    "consecutive_violations": float(stability_guard_consecutive_violations),
                                    "failures": stability_guard_metrics["failures"],
                                }
                            )
                            max_rollbacks = int(stability_guard_metrics["max_rollbacks"])
                            if max_rollbacks > 0 and stability_guard_rollbacks >= max_rollbacks:
                                should_stop_early = bool(stability_guard_metrics["early_stop_after_max_rollbacks"])
                        else:
                            logger.write(
                                {
                                    "phase": phase_name,
                                    "split": "stability_guard",
                                    "step": step,
                                    "guard_action": "violation_without_restore",
                                    "restore_source": "missing",
                                    "rollbacks": float(stability_guard_rollbacks),
                                    "consecutive_violations": float(stability_guard_consecutive_violations),
                                    "failures": stability_guard_metrics["failures"],
                                }
                            )
                else:
                    stability_guard_consecutive_violations = 0
            if guard_passed and val_score > best_val_score:
                best_val_score = val_score
                best_metric_name = metric_name
                best_val = val_metrics["accuracy"]
                best_val_metrics = val_metrics
                save_checkpoint(best_path, model, optimizer, step, {"phase": phase_name})
            save_checkpoint(last_path, model, optimizer, step, {"phase": phase_name})
            if should_stop_early:
                break

    if best_path.exists():
        load_checkpoint(best_path, model)

    phase_wall_time_sec = time.time() - phase_start_time
    eval_batch_size, eval_splits = evaluation_requests(section_cfg=train_cfg, train_cfg=train_cfg)
    final_evals: dict[str, Any] = {}
    for split_name, num_batches in eval_splits:
        split_metrics = evaluate_model(
            model=model,
            benchmark=benchmark,
            device=device,
            benchmark_name=benchmark_name,
            split=split_name,
            num_batches=num_batches,
            batch_size=eval_batch_size,
            route_mode=route_mode,
            compute_penalties=eval_compute_penalties,
            temperature=temperature,
            estimator=estimator,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            routing_cfg=None,
        )
        split_metrics.update({"phase": phase_name, "split": split_name, "step": total_steps})
        logger.write(split_metrics)
        final_evals[split_name] = split_metrics
    if auxiliary_eval_benchmarks:
        selection_eval_final: dict[str, Any] = {}
        for eval_source in auxiliary_eval_benchmarks:
            eval_metrics = evaluate_model(
                model=model,
                benchmark=eval_source.benchmark,
                device=device,
                benchmark_name=eval_source.benchmark_name,
                split=eval_source.split,
                num_batches=eval_source.num_batches,
                batch_size=eval_source.batch_size,
                route_mode=route_mode,
                compute_penalties=eval_compute_penalties,
                temperature=temperature,
                estimator=estimator,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                routing_cfg=None,
            )
            eval_metrics.update({"phase": phase_name, "split": f"selection_eval_{eval_source.name}", "step": total_steps})
            logger.write(eval_metrics)
            selection_eval_final[eval_source.name] = eval_metrics
        final_evals["selection_eval"] = selection_eval_final
    return {
        "best_val_accuracy": best_val,
        "best_val_score": best_val_score,
        "best_metric_name": best_metric_name,
        "best_val_metrics": best_val_metrics,
        **final_evals,
        "checkpoint": str(best_path),
        "phase_wall_time_sec": phase_wall_time_sec,
        "peak_train_memory_mb": peak_train_memory_mb,
        **trainable_summary,
        **partial_init_summary,
        **probe_warmstart_summary,
        **probe_adapter_warmstart_summary,
    }


def run_reinforce_phase(
    *,
    phase_name: str,
    model: PacketRoutingModel,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    results_dir: Path,
    logger: JsonlLogger,
    temperature: float,
) -> dict[str, Any]:
    train_cfg = cfg["training"]
    system_cfg = cfg.get("system", {})
    objective_cfg = cfg["objective"]
    schedule_cfg = cfg.get("objective_schedule", {})
    routing_cfg = cfg.get("routing", {})
    method_cfg = cfg["method"]
    amp_enabled = bool(system_cfg.get("amp", False))
    amp_dtype = str(system_cfg.get("amp_dtype", "bf16"))
    policy_weight = float(method_cfg.get("policy_weight", 1.0))
    supervised_weight = float(method_cfg.get("supervised_weight", 1.0))
    entropy_weight = float(method_cfg.get("entropy_weight", 0.0))

    trainable_summary = configure_trainable_parameters(model, train_cfg)
    optimizer = build_supervised_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    start_step = 0
    best_val = -1.0
    best_path = results_dir / f"{phase_name}_best.pt"
    last_path = results_dir / f"{phase_name}_last.pt"

    resume_path, resume_strict = resolve_resume_checkpoint(cfg)
    if resume_path:
        payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            strict=resume_strict,
        )
        start_step = int(payload.get("step", 0))

    configured_total_steps = int(train_cfg["train_steps"])
    if "train_steps_delta" in train_cfg:
        total_steps = start_step + int(train_cfg["train_steps_delta"])
    else:
        total_steps = configured_total_steps
    eval_compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, total_steps)
    model.train()
    phase_start_time = time.time()
    peak_train_memory_mb = 0.0
    best_val_metrics: dict[str, Any] | None = None
    best_val_score = initial_selection_score(train_cfg)
    best_metric_name = str(train_cfg.get("selection_metric", "accuracy"))
    progress = tqdm(range(start_step, total_steps), disable=False)
    for step in progress:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch = benchmark.sample_batch(
            batch_size=int(train_cfg["batch_size"]),
            split="train",
            step=step,
            device=device,
        )
        train_temperature = current_training_temperature(method_cfg, step)
        compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, step)
        step_routing_cfg = current_routing_cfg(routing_cfg, step)
        forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        memory_payload_targets, memory_payload_mask, memory_payload_weight = build_memory_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        control_targets, control_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        wait_targets, wait_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        release_targets, release_mask, release_weight, release_positive_weight = build_release_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        task_sample_weights = build_task_sample_weights(
            batch,
            train_cfg,
            device=batch.labels.device,
            dtype=batch.observations.dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(
                observations=batch.observations,
                labels=batch.labels,
                route_mode="sample",
                compute_penalties=compute_penalties,
                temperature=train_temperature,
                estimator="straight_through",
                truncate_bptt_steps=int(train_cfg.get("truncate_bptt_steps", 0)),
                detach_prefix_steps=int(train_cfg.get("detach_prefix_steps", 0)),
                late_window_steps=int(train_cfg.get("late_window_steps", 0)),
                forced_actions=forced_actions,
                action_masks=action_masks,
                oracle_actions=oracle_actions,
                oracle_action_mask=oracle_action_mask,
                oracle_route_weight=oracle_route_weight,
                delay_write_targets=batch.delay_write_targets,
                delay_write_mask=batch.delay_write_mask,
                delay_write_weight=delay_write_weight,
                memory_payload_targets=memory_payload_targets,
                memory_payload_mask=memory_payload_mask,
                memory_payload_weight=memory_payload_weight,
                control_targets=control_targets,
                control_mask=control_mask,
                control_weight=control_weight,
                anti_exit_mask=anti_exit_mask,
                anti_exit_weight=anti_exit_weight,
                wait_targets=wait_targets,
                wait_mask=wait_mask,
                wait_weight=wait_weight,
                wait_positive_weight=wait_positive_weight,
                wait_negative_weight=wait_negative_weight,
                release_targets=release_targets,
                release_mask=release_mask,
                release_weight=release_weight,
                release_positive_weight=release_positive_weight,
                task_sample_weights=task_sample_weights,
                final_query_mask=batch.metadata.get("needs_final_query"),
            )
            reward = build_reward(
                output.logits,
                batch.labels,
                output.stats,
                objective_cfg,
                reward_penalties=compute_penalties,
            )
            advantages, baseline = reinforce_advantages(reward, method_cfg)
            policy_logprob = output.stats["policy_logprob"].float()
            policy_loss = -(advantages.detach() * policy_logprob).mean()
            entropy_bonus = output.stats["route_entropy"].float().mean()
            total_loss = (
                supervised_weight * output.loss.float()
                + policy_weight * policy_loss
                - entropy_weight * entropy_bonus
            )
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(train_cfg.get("grad_clip", 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()

        batch_metrics = tensor_dict_mean([summarize_batch(batch, output, benchmark_name)])
        batch_metrics.update(
            {
                "phase": phase_name,
                "split": "train",
                "step": step,
                "examples_per_sec": float(train_cfg["batch_size"]) / max(1e-6, time.time() - start_time),
                "peak_memory_mb": (
                    float(torch.cuda.max_memory_allocated(device) / (1024**2))
                    if device.type == "cuda"
                    else 0.0
                ),
                "temperature": train_temperature,
                "reward_mean": float(reward.mean().detach().item()),
                "reward_std": float(reward.std(unbiased=False).detach().item()),
                "baseline": float(baseline.detach().item()),
                "advantage_mean": float(advantages.mean().detach().item()),
                "advantage_std": float(advantages.std(unbiased=False).detach().item()),
                "policy_logprob_mean": float(policy_logprob.mean().detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "entropy_bonus": float(entropy_bonus.detach().item()),
                "policy_total_loss": float(total_loss.detach().item()),
            }
        )
        peak_train_memory_mb = max(peak_train_memory_mb, float(batch_metrics["peak_memory_mb"]))
        logger.write(batch_metrics)
        progress.set_description(
            f"{phase_name} step={step} reward={batch_metrics['reward_mean']:.3f} acc={batch_metrics['accuracy']:.3f}"
        )

        val_every = int(train_cfg.get("val_every", 25))
        if (step + 1) % val_every == 0 or step + 1 == total_steps:
            val_metrics = evaluate_model(
                model=model,
                benchmark=benchmark,
                device=device,
                benchmark_name=benchmark_name,
                split="val",
                num_batches=int(train_cfg.get("val_batches", 8)),
                batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
                route_mode="hard",
                compute_penalties=eval_compute_penalties,
                temperature=temperature,
                estimator="straight_through",
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                routing_cfg=None,
            )
            val_metrics.update({"phase": phase_name, "split": "val", "step": step})
            peak_train_memory_mb = max(peak_train_memory_mb, float(val_metrics["peak_memory_mb"]))
            logger.write(val_metrics)
            val_score, metric_name = validation_score(val_metrics, cfg, section="training")
            if val_score > best_val_score:
                best_val_score = val_score
                best_metric_name = metric_name
                best_val = val_metrics["accuracy"]
                best_val_metrics = val_metrics
                save_checkpoint(best_path, model, optimizer, step, {"phase": phase_name})
            save_checkpoint(last_path, model, optimizer, step, {"phase": phase_name})

    if best_path.exists():
        load_checkpoint(best_path, model)

    phase_wall_time_sec = time.time() - phase_start_time
    eval_batch_size, eval_splits = evaluation_requests(section_cfg=train_cfg, train_cfg=train_cfg)
    final_evals: dict[str, Any] = {}
    for split_name, num_batches in eval_splits:
        split_metrics = evaluate_model(
            model=model,
            benchmark=benchmark,
            device=device,
            benchmark_name=benchmark_name,
            split=split_name,
            num_batches=num_batches,
            batch_size=eval_batch_size,
            route_mode="hard",
            compute_penalties=eval_compute_penalties,
            temperature=temperature,
            estimator="straight_through",
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            routing_cfg=None,
        )
        split_metrics.update({"phase": phase_name, "split": split_name, "step": total_steps})
        logger.write(split_metrics)
        final_evals[split_name] = split_metrics
    return {
        "best_val_accuracy": best_val,
        "best_val_score": best_val_score,
        "best_metric_name": best_metric_name,
        "best_val_metrics": best_val_metrics,
        **final_evals,
        "checkpoint": str(best_path),
        "phase_wall_time_sec": phase_wall_time_sec,
        "peak_train_memory_mb": peak_train_memory_mb,
        **trainable_summary,
    }


def gather_population_rewards(local_rewards: torch.Tensor, context: DistContext) -> torch.Tensor:
    if not context.enabled:
        return local_rewards
    gathered = [torch.zeros_like(local_rewards) for _ in range(context.world_size)]
    dist.all_gather(gathered, local_rewards)
    return torch.cat(gathered, dim=0)


def run_hybrid_es(
    *,
    model: PacketRoutingModel,
    benchmark,
    benchmark_name: str,
    cfg: dict[str, Any],
    context: DistContext,
    results_dir: Path,
    logger: JsonlLogger,
) -> dict[str, Any]:
    system_cfg = cfg.get("system", {})
    train_cfg = cfg["training"]
    objective_cfg = cfg["objective"]
    schedule_cfg = cfg.get("objective_schedule", {})
    routing_cfg = cfg.get("routing", {})
    es_cfg = cfg["es"]
    amp_enabled = bool(system_cfg.get("amp", False))
    amp_dtype = str(system_cfg.get("amp_dtype", "bf16"))

    warmstart_cfg = cfg.get("warmstart", {})
    warmstart_summary: dict[str, Any] | None = None
    resume_path, resume_strict = resolve_resume_checkpoint(cfg)
    if resume_path:
        load_checkpoint(
            resume_path,
            model,
            strict=resume_strict,
        )
    if warmstart_cfg.get("enabled", False) and context.rank == 0:
        warmstart_summary = run_supervised_phase(
            phase_name="warmstart",
            model=model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg={
                **cfg,
                "training": {
                    **cfg["training"],
                    "train_steps": int(warmstart_cfg.get("train_steps", 50)),
                    "batch_size": int(warmstart_cfg.get("batch_size", train_cfg["batch_size"])),
                    "val_every": int(warmstart_cfg.get("val_every", 25)),
                },
                "routing": {
                    **cfg.get("routing", {}),
                    **warmstart_cfg.get("routing", {}),
                },
                "objective_schedule": {
                    **cfg.get("objective_schedule", {}),
                    **warmstart_cfg.get("objective_schedule", {}),
                },
            },
            device=context.device,
            results_dir=results_dir,
            logger=logger,
            route_mode=str(warmstart_cfg.get("route_mode", "soft")),
            temperature=float(warmstart_cfg.get("temperature", cfg["method"].get("temperature", 1.0))),
            estimator=str(warmstart_cfg.get("estimator", cfg["method"].get("estimator", "straight_through"))),
        )
    if context.enabled:
        distributed_barrier(context)
        warmstart_ckpt = results_dir / "warmstart_best.pt"
        if warmstart_ckpt.exists():
            load_checkpoint(warmstart_ckpt, model)

    es_param_info = configure_es_parameter_names(model, es_cfg)
    parameter_names = es_param_info["es_parameter_names"]

    es = LowRankEvolutionStrategy(
        model=model,
        parameter_names=parameter_names,
        sigma=float(es_cfg["sigma"]),
        rank=int(es_cfg["rank"]),
        lr=float(es_cfg["lr"]),
        weight_decay=float(es_cfg.get("weight_decay", 0.0)),
        noise_reuse=int(es_cfg.get("noise_reuse", 0)),
        optimizer_name=str(es_cfg.get("optimizer", "adam")),
    )

    best_val = -1.0
    best_path = results_dir / "hybrid_es_best.pt"
    generations = int(es_cfg["generations"])
    population = int(es_cfg["population"])
    local_start, local_end = es.local_member_range(population, context.world_size, context.rank)
    phase_start_time = time.time()
    peak_train_memory_mb = 0.0
    best_val_metrics: dict[str, Any] | None = None
    best_val_score = initial_selection_score(es_cfg)
    best_metric_name = str(es_cfg.get("selection_metric", "accuracy"))
    progress = tqdm(range(generations), disable=context.rank != 0)
    eval_compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, generations)

    for generation in progress:
        batch = benchmark.sample_batch(
            batch_size=int(es_cfg.get("batch_size", train_cfg["batch_size"])),
            split="train",
            step=generation,
            device=context.device,
        )
        compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, generation)
        step_routing_cfg = current_routing_cfg(routing_cfg, generation)
        forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        memory_payload_targets, memory_payload_mask, memory_payload_weight = build_memory_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        control_targets, control_mask, control_weight, anti_exit_mask, anti_exit_weight = build_control_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        wait_targets, wait_mask, wait_weight, wait_positive_weight, wait_negative_weight = build_wait_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        release_targets, release_mask, release_weight, release_positive_weight = build_release_controls(
            batch,
            step_routing_cfg,
            split="train",
        )
        local_rewards = []
        local_stats = []
        start_time = time.time()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)

        for member_index in range(local_start, local_end):
            applied = es.perturb_member(generation=generation, member_index=member_index)
            with torch.no_grad():
                with autocast_context(context.device, amp_enabled, amp_dtype):
                    output = model(
                        observations=batch.observations,
                        labels=batch.labels,
                        route_mode="hard",
                        compute_penalties=compute_penalties,
                        temperature=float(cfg["method"].get("temperature", 1.0)),
                        estimator="straight_through",
                        truncate_bptt_steps=0,
                        forced_actions=forced_actions,
                        action_masks=action_masks,
                        oracle_actions=oracle_actions,
                        oracle_action_mask=oracle_action_mask,
                        oracle_route_weight=oracle_route_weight,
                        delay_write_targets=batch.delay_write_targets,
                        delay_write_mask=batch.delay_write_mask,
                        delay_write_weight=delay_write_weight,
                        memory_payload_targets=memory_payload_targets,
                        memory_payload_mask=memory_payload_mask,
                        memory_payload_weight=memory_payload_weight,
                        control_targets=control_targets,
                        control_mask=control_mask,
                        control_weight=control_weight,
                        anti_exit_mask=anti_exit_mask,
                        anti_exit_weight=anti_exit_weight,
                        wait_targets=wait_targets,
                        wait_mask=wait_mask,
                        wait_weight=wait_weight,
                        wait_positive_weight=wait_positive_weight,
                        wait_negative_weight=wait_negative_weight,
                        release_targets=release_targets,
                        release_mask=release_mask,
                        release_weight=release_weight,
                        release_positive_weight=release_positive_weight,
                        final_query_mask=batch.metadata.get("needs_final_query"),
                    )
                reward = build_reward(
                    output.logits,
                    batch.labels,
                    output.stats,
                    objective_cfg,
                    reward_penalties=compute_penalties,
                ).mean()
                local_rewards.append(reward)
                local_stats.append(summarize_batch(batch, output, benchmark_name))
            es.revert_member(applied)

        reward_tensor = torch.stack(local_rewards)
        all_rewards = gather_population_rewards(reward_tensor, context)
        if context.rank == 0:
            fitness_mode = str(es_cfg.get("fitness_mode", "standardize"))
            if fitness_mode == "centered_rank":
                fitness = centered_rank_fitness(all_rewards).to(context.device)
            else:
                fitness = standardize_fitness(all_rewards).to(context.device)
            updates = es.compute_updates(generation=generation, fitness=fitness)
            es.apply_updates(updates)
        if context.enabled:
            es.broadcast_parameters(src=0)
            distributed_barrier(context)

        if context.rank == 0:
            merged = tensor_dict_mean(local_stats)
            merged.update(
                {
                    "phase": "hybrid_es",
                    "split": "train",
                    "generation": generation,
                    "reward": float(all_rewards.mean().item()),
                    "reward_std": float(all_rewards.std(unbiased=False).item()),
                    "reward_min": float(all_rewards.min().item()),
                    "reward_max": float(all_rewards.max().item()),
                    "population_per_sec": population / max(1e-6, time.time() - start_time),
                    "sigma": float(es.sigma),
                    "es_rank": int(es.rank),
                    "population": int(population),
                    "peak_memory_mb": (
                        float(torch.cuda.max_memory_allocated(context.device) / (1024**2))
                        if context.device.type == "cuda"
                        else 0.0
                    ),
                }
            )
            peak_train_memory_mb = max(peak_train_memory_mb, float(merged["peak_memory_mb"]))
            logger.write(merged)
            progress.set_description(
                f"hybrid_es gen={generation} reward={merged['reward']:.3f} acc={merged['accuracy']:.3f}"
            )

            val_every = int(es_cfg.get("val_every", 10))
            if (generation + 1) % val_every == 0 or generation + 1 == generations:
                val_metrics = evaluate_model(
                    model=model,
                    benchmark=benchmark,
                    device=context.device,
                    benchmark_name=benchmark_name,
                    split="val",
                    num_batches=int(es_cfg.get("val_batches", train_cfg.get("val_batches", 8))),
                    batch_size=int(es_cfg.get("eval_batch_size", train_cfg.get("val_batch_size", train_cfg["batch_size"]))),
                    route_mode="hard",
                    compute_penalties=eval_compute_penalties,
                    temperature=float(cfg["method"].get("temperature", 1.0)),
                    estimator="straight_through",
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    routing_cfg=None,
                )
                val_metrics.update({"phase": "hybrid_es", "split": "val", "generation": generation})
                peak_train_memory_mb = max(peak_train_memory_mb, float(val_metrics["peak_memory_mb"]))
                logger.write(val_metrics)
                val_score, metric_name = validation_score(val_metrics, cfg, section="es")
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_metric_name = metric_name
                    best_val = val_metrics["accuracy"]
                    best_val_metrics = val_metrics
                    save_checkpoint(best_path, model, None, generation, {"phase": "hybrid_es"})

    if context.rank == 0 and best_path.exists():
        load_checkpoint(best_path, model)

    if context.enabled:
        distributed_barrier(context)
    eval_batch_size, eval_splits = evaluation_requests(section_cfg=es_cfg, train_cfg=train_cfg)
    final_evals: dict[str, Any] = {}
    if context.rank == 0:
        for split_name, num_batches in eval_splits:
            split_metrics = evaluate_model(
                model=model,
                benchmark=benchmark,
                device=context.device,
                benchmark_name=benchmark_name,
                split=split_name,
                num_batches=num_batches,
                batch_size=eval_batch_size,
                route_mode="hard",
                compute_penalties=eval_compute_penalties,
                temperature=float(cfg["method"].get("temperature", 1.0)),
                estimator="straight_through",
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                routing_cfg=None,
            )
            split_metrics.update({"phase": "hybrid_es", "split": split_name, "generation": generations})
            logger.write(split_metrics)
            final_evals[split_name] = split_metrics
    total_wall_time_sec = time.time() - phase_start_time
    return {
        "warmstart": warmstart_summary,
        "best_val_accuracy": best_val,
        "best_val_score": best_val_score,
        "best_metric_name": best_metric_name,
        "best_val_metrics": best_val_metrics,
        **final_evals,
        "parameter_names": parameter_names,
        "parameter_count": sum(dict(model.named_parameters())[name].numel() for name in parameter_names),
        "es_wall_time_sec": total_wall_time_sec,
        "total_wall_time_sec": total_wall_time_sec + float((warmstart_summary or {}).get("phase_wall_time_sec", 0.0)),
        "peak_train_memory_mb": peak_train_memory_mb,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    system_cfg = cfg.get("system", {})
    cpu_threads = int(system_cfg.get("cpu_threads", 16))
    torch.set_num_threads(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

    context = setup_distributed()
    seed_everything(int(cfg["experiment"]["seed"]) + context.rank)
    results_dir = create_results_dir(cfg, args.results_dir)
    if context.rank == 0:
        (results_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg["config_path"], results_dir / "config.yaml")
        save_json(
            results_dir / "system_info.json",
            {
                "rank": context.rank,
                "world_size": context.world_size,
                "device": str(context.device),
                "cpu_threads": cpu_threads,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "logical_cpus": psutil.cpu_count(logical=True),
            },
        )

    benchmark = build_benchmark(cfg["benchmark"])
    model_cfg = benchmark_model_config(cfg["model"], benchmark)
    model = PacketRoutingModel(model_cfg).to(context.device)
    benchmark_name = cfg["benchmark"]["name"]
    logger = JsonlLogger(results_dir / "metrics.jsonl")

    summary: dict[str, Any]
    method_name = cfg["method"]["name"]
    if context.enabled and method_name != "hybrid_es":
        raise ValueError("Distributed execution is only supported for hybrid_es in this implementation.")

    if method_name == "soft":
        summary = run_supervised_phase(
            phase_name="soft",
            model=model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg=cfg,
            device=context.device,
            results_dir=results_dir,
            logger=logger,
            route_mode="soft",
            temperature=float(cfg["method"].get("temperature", 1.0)),
            estimator="straight_through",
        )
    elif method_name == "hard_st":
        summary = run_supervised_phase(
            phase_name="hard_st",
            model=model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg=cfg,
            device=context.device,
            results_dir=results_dir,
            logger=logger,
            route_mode="hard_st",
            temperature=float(cfg["method"].get("temperature", 1.0)),
            estimator=str(cfg["method"].get("estimator", "straight_through")),
        )
    elif method_name == "hybrid_es":
        summary = run_hybrid_es(
            model=model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg=cfg,
            context=context,
            results_dir=results_dir,
            logger=logger,
        )
    elif method_name == "reinforce":
        summary = run_reinforce_phase(
            phase_name="reinforce",
            model=model,
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            cfg=cfg,
            device=context.device,
            results_dir=results_dir,
            logger=logger,
            temperature=float(cfg["method"].get("temperature", 1.0)),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    if context.rank == 0:
        save_json(
            results_dir / "summary.json",
            {
                "config_path": cfg["config_path"],
                "benchmark": benchmark_name,
                "method": method_name,
                "results_dir": str(results_dir),
                "summary": summary,
            },
        )
    cleanup_distributed(context)


if __name__ == "__main__":
    main()
