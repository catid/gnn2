from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from src.data import build_benchmark
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a synthetic benchmark configuration.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", required=True, help="Output JSON path.")
    return parser.parse_args()


def heuristic_decode_accuracy(name: str, batch) -> float | None:
    obs = batch.observations.cpu()
    labels = batch.labels.cpu()
    if name in {"long_horizon_memory", "long_horizon_memory_v1"}:
        trigger = obs[:, :, 0, 0] > 0.5
        payload = obs[:, :, 0, 2:6].argmax(dim=-1)
        query = obs[:, -1, 0, 6:10].argmax(dim=-1)
        trigger_time = trigger.float().argmax(dim=-1)
        selected_payload = payload[torch.arange(payload.shape[0]), trigger_time]
        pred = torch.remainder(selected_payload + query, 4)
        return float((pred == labels).float().mean().item())

    if name == "long_horizon_memory_v2":
        modes = batch.modes.cpu()
        trigger = obs[:, :, 0, 0] > 0.5
        payload = obs[:, :, 0, 5:9].argmax(dim=-1)
        query = obs[:, -1, 0, 9:13].argmax(dim=-1)
        trigger_time = trigger.float().argmax(dim=-1)
        selected_payload = payload[torch.arange(payload.shape[0]), trigger_time]
        pred = torch.zeros_like(labels)
        easy_mask = modes == 0
        trigger_exit_mask = modes == 1
        final_mask = modes == 2
        pred[easy_mask] = payload[easy_mask, 0]
        pred[trigger_exit_mask] = selected_payload[trigger_exit_mask]
        pred[final_mask] = torch.remainder(selected_payload[final_mask] + query[final_mask], 4)
        return float((pred == labels).float().mean().item())

    return None


def early_only_accuracy(name: str, batch) -> float | None:
    obs = batch.observations.cpu()
    labels = batch.labels.cpu()
    if name in {"long_horizon_memory", "long_horizon_memory_v1"}:
        pred = torch.zeros_like(labels)
        return float((pred == labels).float().mean().item())
    if name == "long_horizon_memory_v2":
        payload0 = obs[:, 0, 0, 5:9].argmax(dim=-1)
        pred = payload0
        return float((pred == labels).float().mean().item())
    return None


def final_only_accuracy(name: str, batch) -> float | None:
    obs = batch.observations.cpu()
    labels = batch.labels.cpu()
    if name in {"long_horizon_memory", "long_horizon_memory_v1"}:
        query = obs[:, -1, 0, 6:10].argmax(dim=-1)
        pred = query % 4
        return float((pred == labels).float().mean().item())
    if name == "long_horizon_memory_v2":
        query = obs[:, -1, 0, 9:13].argmax(dim=-1)
        pred = query % 4
        return float((pred == labels).float().mean().item())
    return None


def unique_route_patterns(batch) -> int:
    if batch.oracle_actions is None or batch.oracle_action_mask is None:
        return 0
    patterns = set()
    actions = batch.oracle_actions.cpu()
    masks = batch.oracle_action_mask.cpu()
    for i in range(actions.shape[0]):
        valid = masks[i] > 0.0
        patterns.add(tuple(actions[i][valid].tolist()))
    return len(patterns)


def summarize_benchmark(cfg: dict[str, Any], split: str, batches: int, batch_size: int) -> dict[str, Any]:
    benchmark_cfg = cfg["benchmark"]
    name = benchmark_cfg["name"]
    benchmark = build_benchmark(benchmark_cfg)
    objective = cfg.get("objective", {})

    label_counter: Counter[int] = Counter()
    mode_counter: Counter[int] = Counter()
    trigger_times: list[torch.Tensor] = []
    retrieval_distances: list[torch.Tensor] = []
    oracle_delays: list[torch.Tensor] = []
    oracle_exit_times: list[torch.Tensor] = []
    heuristic_accs: list[float] = []
    early_accs: list[float] = []
    final_accs: list[float] = []
    unique_patterns = 0

    for step in range(batches):
        batch = benchmark.sample_batch(batch_size=batch_size, split=split, step=step, device="cpu")
        label_counter.update(batch.labels.tolist())
        mode_counter.update(batch.modes.tolist())
        oracle_delays.append(batch.oracle_delays.float())
        oracle_exit_times.append(batch.oracle_exit_time.float())
        unique_patterns = max(unique_patterns, unique_route_patterns(batch))
        if "trigger_time" in batch.metadata:
            trigger_times.append(batch.metadata["trigger_time"].float())
        if "retrieval_distance" in batch.metadata:
            retrieval_distances.append(batch.metadata["retrieval_distance"].float())
        value = heuristic_decode_accuracy(name, batch)
        if value is not None:
            heuristic_accs.append(value)
        value = early_only_accuracy(name, batch)
        if value is not None:
            early_accs.append(value)
        value = final_only_accuracy(name, batch)
        if value is not None:
            final_accs.append(value)

    delay_values = torch.cat(oracle_delays) if oracle_delays else torch.zeros(1)
    exit_values = torch.cat(oracle_exit_times) if oracle_exit_times else torch.zeros(1)
    trigger_values = torch.cat(trigger_times) if trigger_times else torch.zeros(1)
    retrieval_values = torch.cat(retrieval_distances) if retrieval_distances else torch.zeros(1)
    lambda_delay = float(objective.get("lambda_delay", 0.0))
    chance_ce = math.log(int(benchmark_cfg["num_classes"]))
    mean_delay_penalty = lambda_delay * float(delay_values.mean().item())
    reward_margin_if_perfect = chance_ce - mean_delay_penalty

    mode_names = getattr(benchmark, "mode_names", {}) or {}
    return {
        "benchmark_name": name,
        "num_classes": int(benchmark_cfg["num_classes"]),
        "seq_len": int(benchmark_cfg["seq_len"]),
        "batches": batches,
        "batch_size": batch_size,
        "mode_names": {str(key): value for key, value in mode_names.items()},
        "label_histogram": dict(sorted(label_counter.items())),
        "mode_histogram": {
            mode_names.get(mode_id, str(mode_id)): count
            for mode_id, count in sorted(mode_counter.items())
        },
        "mean_oracle_delays": float(delay_values.mean().item()),
        "mean_oracle_exit_time": float(exit_values.mean().item()),
        "mean_trigger_time": float(trigger_values.mean().item()),
        "mean_retrieval_distance": float(retrieval_values.mean().item()),
        "max_unique_oracle_route_patterns_per_batch": unique_patterns,
        "heuristic_full_decode_accuracy": float(sum(heuristic_accs) / max(1, len(heuristic_accs))),
        "early_only_accuracy": float(sum(early_accs) / max(1, len(early_accs))),
        "final_only_accuracy": float(sum(final_accs) / max(1, len(final_accs))),
        "lambda_delay": lambda_delay,
        "mean_delay_penalty": mean_delay_penalty,
        "chance_cross_entropy": chance_ce,
        "break_even_ce_for_delay_vs_immediate_exit": chance_ce - mean_delay_penalty,
        "reward_margin_if_perfect_delay_policy": reward_margin_if_perfect,
        "delay_is_objectively_plausible": reward_margin_if_perfect > 0.0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary = summarize_benchmark(
        cfg,
        split=args.split,
        batches=args.batches,
        batch_size=args.batch_size,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
