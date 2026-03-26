from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data import BenchmarkBatch, build_benchmark
from src.models import PacketRoutingModel
from src.train.run import (
    autocast_context,
    benchmark_model_config,
    current_compute_penalties,
    load_checkpoint,
    seed_everything,
)
from src.utils.config import load_config
from src.utils.phase12_verify import infer_method_settings, resolve_checkpoint, resolve_run_config


@dataclass
class LoadedRun:
    name: str
    run_dir: Path
    cfg: dict[str, Any]
    model: PacketRoutingModel
    route_mode: str
    temperature: float
    estimator: str
    compute_penalties: dict[str, float]
    amp_enabled: bool
    amp_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase-14 content-failure hard slices on saved checkpoints.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory. Pass multiple times.")
    parser.add_argument("--name", action="append", default=[], help="Optional display names matching --run-dir order.")
    parser.add_argument("--eval-config", required=True, help="Evaluation config for deterministic hard-slice generation.")
    parser.add_argument("--split", default="confirm", help="Benchmark split to evaluate.")
    parser.add_argument("--num-batches", type=int, default=32, help="Number of deterministic batches.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--late-slack", type=float, default=8.0, help="Relaxed late-route exit slack from query_time.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    return parser.parse_args()


def load_run(run_dir: Path, eval_cfg: dict[str, Any], device: torch.device, name: str) -> LoadedRun:
    cfg, payload = resolve_run_config(run_dir)
    method_name, route_mode, temperature, estimator, total_steps, _ = infer_method_settings(cfg)
    checkpoint_path = resolve_checkpoint(run_dir, method_name, None)
    model = PacketRoutingModel(benchmark_model_config(cfg["model"], build_benchmark(eval_cfg["benchmark"]))).to(device)
    load_checkpoint(checkpoint_path, model, strict=bool(cfg.get("resume_strict", True)))
    model.eval()
    compute_penalties = current_compute_penalties(cfg["objective"], cfg.get("objective_schedule", {}), total_steps)
    system_cfg = cfg.get("system", {})
    return LoadedRun(
        name=name,
        run_dir=run_dir,
        cfg=cfg,
        model=model,
        route_mode=route_mode,
        temperature=temperature,
        estimator=estimator,
        compute_penalties=compute_penalties,
        amp_enabled=bool(system_cfg.get("amp", False)),
        amp_dtype=str(system_cfg.get("amp_dtype", "bf16")),
    )


def sample_rows(
    *,
    run: LoadedRun,
    benchmark,
    benchmark_name: str,
    split: str,
    num_batches: int,
    batch_size: int,
    device: torch.device,
    late_slack: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(num_batches):
            batch: BenchmarkBatch = benchmark.sample_batch(batch_size=batch_size, split=split, step=step, device=device)
            with autocast_context(device, run.amp_enabled, run.amp_dtype):
                output = run.model(
                    observations=batch.observations,
                    labels=batch.labels,
                    route_mode=run.route_mode,
                    compute_penalties=run.compute_penalties,
                    temperature=run.temperature,
                    estimator=run.estimator,
                    truncate_bptt_steps=0,
                    final_query_mask=batch.metadata.get("needs_final_query"),
                    return_trace=True,
                )

            predictions = output.logits.argmax(dim=-1)
            rounded_hops = torch.round(output.stats["hops"]).long()
            rounded_delays = torch.round(output.stats["delays"]).long()
            rounded_exit = torch.round(output.stats["exit_time"]).long()
            route_match = (
                (rounded_hops == batch.oracle_hops)
                & (rounded_delays == batch.oracle_delays)
                & (rounded_exit == batch.oracle_exit_time)
            )
            query_time = batch.metadata.get("query_time")
            if query_time is None:
                query_time = torch.zeros_like(output.stats["exit_time"]).long()
            needs_final_query = batch.metadata.get("needs_final_query")
            if needs_final_query is None:
                needs_final_query = torch.zeros_like(batch.labels)
            retrieval_distance = batch.metadata.get("retrieval_distance")
            if retrieval_distance is None:
                retrieval_distance = torch.zeros_like(batch.labels)
            payload = batch.metadata.get("payload")
            if payload is None:
                payload = torch.full_like(batch.labels, -1)
            query = batch.metadata.get("query")
            if query is None:
                query = torch.full_like(batch.labels, -1)
            trigger_time = batch.metadata.get("trigger_time")
            if trigger_time is None:
                trigger_time = torch.full_like(batch.labels, -1)

            late_relaxed = (output.stats["exit_time"] >= (query_time.float() - late_slack)) & (needs_final_query > 0)
            late_strict = route_match & (needs_final_query > 0)

            for idx in range(batch.labels.shape[0]):
                rows.append(
                    {
                        "sample_id": f"{split}:{step}:{idx}",
                        "benchmark_name": benchmark_name,
                        "run_name": run.name,
                        "step": step,
                        "sample_index": idx,
                        "label": int(batch.labels[idx].item()),
                        "prediction": int(predictions[idx].item()),
                        "correct": bool(predictions[idx].item() == batch.labels[idx].item()),
                        "mode": int(batch.modes[idx].item()),
                        "needs_final_query": bool(needs_final_query[idx].item() > 0),
                        "route_match": bool(route_match[idx].item()),
                        "exit_time": float(output.stats["exit_time"][idx].item()),
                        "query_time": float(query_time[idx].item()),
                        "late_relaxed": bool(late_relaxed[idx].item()),
                        "late_strict": bool(late_strict[idx].item()),
                        "retrieval_distance": int(retrieval_distance[idx].item()),
                        "payload": int(payload[idx].item()),
                        "query": int(query[idx].item()),
                        "trigger_time": int(trigger_time[idx].item()),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _acc(filtered: list[dict[str, Any]]) -> float:
        if not filtered:
            return float("nan")
        return sum(1.0 for row in filtered if row["correct"]) / len(filtered)

    dqf_rows = [row for row in rows if row["needs_final_query"]]
    late_relaxed_rows = [row for row in dqf_rows if row["late_relaxed"]]
    late_strict_rows = [row for row in dqf_rows if row["late_strict"]]
    if dqf_rows:
        distances = sorted(int(row["retrieval_distance"]) for row in dqf_rows)
        q3 = distances[(3 * (len(distances) - 1)) // 4]
    else:
        q3 = 0
    hard_distance_rows = [row for row in late_relaxed_rows if int(row["retrieval_distance"]) >= q3]
    wrong_late_rows = [row for row in late_relaxed_rows if not row["correct"]]
    wrong_late_strict_rows = [row for row in late_strict_rows if not row["correct"]]
    return {
        "counts": {
            "all": len(rows),
            "delay_to_final_query": len(dqf_rows),
            "late_relaxed": len(late_relaxed_rows),
            "late_strict": len(late_strict_rows),
            "late_wrong_content": len(wrong_late_rows),
            "late_wrong_content_strict": len(wrong_late_strict_rows),
            "late_hard_distance": len(hard_distance_rows),
        },
        "accuracy": {
            "all": _acc(rows),
            "delay_to_final_query": _acc(dqf_rows),
            "late_relaxed": _acc(late_relaxed_rows),
            "late_strict": _acc(late_strict_rows),
            "late_hard_distance": _acc(hard_distance_rows),
        },
        "retrieval_distance_q3": q3,
    }


def pairwise_summary(
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_by_id = {row["sample_id"]: row for row in first_rows}
    second_by_id = {row["sample_id"]: row for row in second_rows}
    shared_ids = sorted(first_by_id.keys() & second_by_id.keys())

    disagreement_late = []
    second_beats_first = []
    first_beats_second = []
    for sample_id in shared_ids:
        first = first_by_id[sample_id]
        second = second_by_id[sample_id]
        both_late = first["late_relaxed"] and second["late_relaxed"]
        if both_late and first["prediction"] != second["prediction"]:
            disagreement_late.append(sample_id)
            if second["correct"] and not first["correct"]:
                second_beats_first.append(sample_id)
            if first["correct"] and not second["correct"]:
                first_beats_second.append(sample_id)
    return {
        "shared_examples": len(shared_ids),
        "late_relaxed_disagreements": len(disagreement_late),
        "second_beats_first_on_late_disagreements": len(second_beats_first),
        "first_beats_second_on_late_disagreements": len(first_beats_second),
        "late_relaxed_disagreement_sample_ids": disagreement_late[:256],
    }


def main() -> None:
    args = parse_args()
    eval_cfg = load_config(args.eval_config)
    benchmark = build_benchmark(eval_cfg["benchmark"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = [Path(item) for item in args.run_dir]
    names = args.name if args.name else [path.name for path in run_dirs]
    if len(names) != len(run_dirs):
        raise ValueError("--name must be omitted or provided once per --run-dir.")

    seed_everything(int(eval_cfg.get("experiment", {}).get("seed", 0)))
    loaded_runs = [load_run(path, eval_cfg, device, name) for path, name in zip(run_dirs, names, strict=True)]

    by_run_rows: dict[str, list[dict[str, Any]]] = {}
    for run in loaded_runs:
        by_run_rows[run.name] = sample_rows(
            run=run,
            benchmark=benchmark,
            benchmark_name=eval_cfg["benchmark"]["name"],
            split=args.split,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            device=device,
            late_slack=args.late_slack,
        )

    rows_path = Path(args.out).with_suffix(".rows.csv")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "benchmark_name",
        "run_name",
        "step",
        "sample_index",
        "label",
        "prediction",
        "correct",
        "mode",
        "needs_final_query",
        "route_match",
        "exit_time",
        "query_time",
        "late_relaxed",
        "late_strict",
        "retrieval_distance",
        "payload",
        "query",
        "trigger_time",
    ]
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_name in names:
            for row in by_run_rows[run_name]:
                writer.writerow(row)

    payload: dict[str, Any] = {
        "eval_config": args.eval_config,
        "split": args.split,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "late_slack": args.late_slack,
        "rows_csv": str(rows_path),
        "runs": {},
        "pairwise": {},
    }
    for run_name in names:
        payload["runs"][run_name] = summarize_rows(by_run_rows[run_name])
    if len(names) >= 2:
        payload["pairwise"][f"{names[0]}__vs__{names[1]}"] = pairwise_summary(
            by_run_rows[names[0]],
            by_run_rows[names[1]],
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out_path), "rows": str(rows_path)}, indent=2))


if __name__ == "__main__":
    main()
