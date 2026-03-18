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
from typing import Any

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from src.data import BenchmarkBatch, build_benchmark
from src.data.benchmarks import LONG_MEMORY_MODE_DELAY_TO_TRIGGER_EXIT
from src.es import LowRankEvolutionStrategy, standardize_fitness
from src.models import ACTION_EXIT, PacketRoutingModel
from src.utils.config import load_config


@dataclass
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run packet-routing experiments.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--results-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume.")
    return parser.parse_args()


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
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
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
        "early_exit_rate",
        "exit_rate",
        "exit_time",
        "packet_age_mean",
        "route_entropy",
        "router_confidence",
        "route_match",
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


def save_checkpoint(path: Path, model: PacketRoutingModel, optimizer: torch.optim.Optimizer | None, step: int, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: PacketRoutingModel, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    start_step = 0
    best_val = -1.0
    best_path = results_dir / f"{phase_name}_best.pt"
    last_path = results_dir / f"{phase_name}_last.pt"

    if cfg.get("resume"):
        payload = load_checkpoint(cfg["resume"], model, optimizer)
        start_step = int(payload.get("step", 0))

    total_steps = int(steps_override or train_cfg["train_steps"])
    eval_compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, total_steps)
    model.train()
    phase_start_time = time.time()
    peak_train_memory_mb = 0.0
    best_val_metrics: dict[str, Any] | None = None
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
        compute_penalties = current_compute_penalties(objective_cfg, schedule_cfg, step)
        forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
            batch,
            routing_cfg,
            split="train",
        )
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        with autocast_context(device, amp_enabled, amp_dtype):
            output = model(
                observations=batch.observations,
                labels=batch.labels,
                route_mode=route_mode,
                compute_penalties=compute_penalties,
                temperature=temperature,
                estimator=estimator,
                truncate_bptt_steps=int(train_cfg.get("truncate_bptt_steps", 0)),
                forced_actions=forced_actions,
                action_masks=action_masks,
                oracle_actions=oracle_actions,
                oracle_action_mask=oracle_action_mask,
                oracle_route_weight=oracle_route_weight,
                delay_write_targets=batch.delay_write_targets,
                delay_write_mask=batch.delay_write_mask,
                delay_write_weight=delay_write_weight,
            )
        scaler.scale(output.loss).backward()
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
            val_metrics.update({"phase": phase_name, "split": "val", "step": step})
            peak_train_memory_mb = max(peak_train_memory_mb, float(val_metrics["peak_memory_mb"]))
            logger.write(val_metrics)
            if val_metrics["accuracy"] > best_val:
                best_val = val_metrics["accuracy"]
                best_val_metrics = val_metrics
                save_checkpoint(best_path, model, optimizer, step, {"phase": phase_name})
            save_checkpoint(last_path, model, optimizer, step, {"phase": phase_name})

    if best_path.exists():
        load_checkpoint(best_path, model)

    phase_wall_time_sec = time.time() - phase_start_time
    test_metrics = evaluate_model(
        model=model,
        benchmark=benchmark,
        device=device,
        benchmark_name=benchmark_name,
        split="test",
        num_batches=int(train_cfg.get("test_batches", 16)),
        batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
        route_mode=route_mode,
        compute_penalties=eval_compute_penalties,
        temperature=temperature,
        estimator=estimator,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        routing_cfg=None,
    )
    test_metrics.update({"phase": phase_name, "split": "test", "step": total_steps})
    logger.write(test_metrics)
    return {
        "best_val_accuracy": best_val,
        "best_val_metrics": best_val_metrics,
        "test": test_metrics,
        "checkpoint": str(best_path),
        "phase_wall_time_sec": phase_wall_time_sec,
        "peak_train_memory_mb": peak_train_memory_mb,
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

    parameter_names = model.es_parameter_names(include_adapters=bool(es_cfg.get("evolve_adapters", False)))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in parameter_names

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
        forced_actions, action_masks, oracle_actions, oracle_action_mask, oracle_route_weight, delay_write_weight = build_routing_controls(
            batch,
            routing_cfg,
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
                if val_metrics["accuracy"] > best_val:
                    best_val = val_metrics["accuracy"]
                    best_val_metrics = val_metrics
                    save_checkpoint(best_path, model, None, generation, {"phase": "hybrid_es"})

    if context.rank == 0 and best_path.exists():
        load_checkpoint(best_path, model)

    if context.enabled:
        distributed_barrier(context)
    test_metrics = None
    if context.rank == 0:
        test_metrics = evaluate_model(
            model=model,
            benchmark=benchmark,
            device=context.device,
            benchmark_name=benchmark_name,
            split="test",
            num_batches=int(es_cfg.get("test_batches", train_cfg.get("test_batches", 16))),
            batch_size=int(es_cfg.get("eval_batch_size", train_cfg.get("val_batch_size", train_cfg["batch_size"]))),
            route_mode="hard",
            compute_penalties=eval_compute_penalties,
            temperature=float(cfg["method"].get("temperature", 1.0)),
            estimator="straight_through",
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            routing_cfg=None,
        )
        test_metrics.update({"phase": "hybrid_es", "split": "test", "generation": generations})
        logger.write(test_metrics)
    total_wall_time_sec = time.time() - phase_start_time
    return {
        "warmstart": warmstart_summary,
        "best_val_accuracy": best_val,
        "best_val_metrics": best_val_metrics,
        "test": test_metrics,
        "parameter_names": parameter_names,
        "parameter_count": sum(dict(model.named_parameters())[name].numel() for name in parameter_names),
        "es_wall_time_sec": total_wall_time_sec,
        "total_wall_time_sec": total_wall_time_sec + float((warmstart_summary or {}).get("phase_wall_time_sec", 0.0)),
        "peak_train_memory_mb": peak_train_memory_mb,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["resume"] = args.resume

    system_cfg = cfg.get("system", {})
    cpu_threads = int(system_cfg.get("cpu_threads", 16))
    torch.set_num_threads(cpu_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

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
    model_cfg = {
        **cfg["model"],
        "num_nodes": benchmark.num_nodes,
        "obs_dim": benchmark.obs_dim,
        "num_classes": benchmark.num_classes,
        "max_total_steps": benchmark.config.get(
            "max_total_steps",
            benchmark.config.get("seq_len", 2) * max(benchmark.num_nodes, 1) * 2,
        ),
    }
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
