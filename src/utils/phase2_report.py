from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.utils.config import load_config


METHOD_LABELS = {
    "soft": "Soft",
    "hard_st": "Hard ST",
    "hybrid_es": "Hybrid ES",
}

METHOD_COLORS = {
    "soft": "#1b9e77",
    "hard_st": "#d95f02",
    "hybrid_es": "#7570b3",
}

BENCHMARK_LABELS = {
    "long_horizon_memory_v1": "Benchmark B v1",
    "long_horizon_memory_v2": "Benchmark B v2",
}


@dataclass
class RunRecord:
    run_dir: Path
    benchmark: str
    method: str
    config: dict[str, Any]
    summary: dict[str, Any]
    test: dict[str, Any]

    @property
    def run_name(self) -> str:
        return self.run_dir.name

    @property
    def benchmark_label(self) -> str:
        return BENCHMARK_LABELS.get(self.benchmark, self.benchmark)

    @property
    def method_label(self) -> str:
        return METHOD_LABELS.get(self.method, self.method)

    @property
    def seq_len(self) -> int:
        return int(self.config.get("benchmark", {}).get("seq_len", 0))

    @property
    def accuracy(self) -> float:
        return float(self.test.get("accuracy", float("nan")))

    @property
    def compute(self) -> float:
        return float(self.test.get("compute", float("nan")))

    @property
    def delay_rate(self) -> float:
        return float(self.test.get("delay_rate", float("nan")))

    @property
    def route_match(self) -> float:
        return float(self.test.get("route_match", float("nan")))

    @property
    def early_exit_rate(self) -> float:
        return float(self.test.get("early_exit_rate", float("nan")))

    @property
    def wall_time_sec(self) -> float:
        if "phase_wall_time_sec" in self.summary:
            return float(self.summary["phase_wall_time_sec"])
        return float("nan")

    @property
    def peak_train_memory_mb(self) -> float:
        peak = self.summary.get("peak_train_memory_mb")
        if peak is not None:
            return float(peak)
        return float(self.test.get("peak_memory_mb", float("nan")))

    @property
    def family_label(self) -> str:
        parts: list[str] = []
        model_cfg = self.config.get("model", {})
        packet_update = str(model_cfg.get("packet_update", "residual"))
        delay_state_mode = str(model_cfg.get("delay_state_mode", "updated"))
        routing_cfg = self.config.get("routing", {})
        if packet_update != "residual":
            parts.append(packet_update)
        if delay_state_mode != "updated":
            parts.append(delay_state_mode)
        if routing_cfg.get("force_oracle_actions"):
            parts.append("oracle-route")
        if routing_cfg.get("exit_mask_final_query_only") or routing_cfg.get("exit_mask_trigger_exit_until_trigger"):
            parts.append("mask-curriculum")
        if float(routing_cfg.get("delay_write_weight", 0.0)) > 0.0:
            parts.append("write-aux")
        if self.method == "hybrid_es":
            es_cfg = self.config.get("es", {})
            parts.append(f"pop{int(es_cfg.get('population', 0))}")
            parts.append(f"r{int(es_cfg.get('rank', 0))}")
            if es_cfg.get("evolve_adapters"):
                parts.append("adapters")
        return ", ".join(parts) if parts else "baseline"


@dataclass
class AuditRecord:
    path: Path
    payload: dict[str, Any]

    @property
    def benchmark(self) -> str:
        path_text = str(self.path).lower()
        if "benchmark_b_v2" in path_text:
            return "long_horizon_memory_v2"
        return "long_horizon_memory_v1"

    @property
    def benchmark_label(self) -> str:
        return BENCHMARK_LABELS.get(self.benchmark, self.benchmark)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the phase-2 benchmark report.")
    parser.add_argument("--results-dir", action="append", required=True, help="Results root or run dir. Repeatable.")
    parser.add_argument("--out", required=True, help="Markdown output path.")
    return parser.parse_args()


def find_run_dirs(results_dirs: list[Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for root in results_dirs:
        if (root / "summary.json").exists():
            run_dirs.add(root)
            continue
        for path in root.rglob("summary.json"):
            run_dirs.add(path.parent)
    return sorted(run_dirs)


def find_audits(results_dirs: list[Path]) -> list[Path]:
    audit_paths: set[Path] = set()
    for root in results_dirs:
        if (root / "audit.json").exists():
            audit_paths.add(root / "audit.json")
            continue
        for path in root.rglob("audit.json"):
            audit_paths.add(path)
    return sorted(audit_paths)


def load_run(run_dir: Path) -> RunRecord:
    payload = json.loads((run_dir / "summary.json").read_text())
    config = load_config(payload["config_path"])
    summary = payload["summary"]
    test = summary.get("test", {})
    return RunRecord(
        run_dir=run_dir,
        benchmark=payload["benchmark"],
        method=payload["method"],
        config=config,
        summary=summary,
        test=test,
    )


def load_metrics(run: RunRecord) -> list[dict[str, Any]]:
    metrics_path = run.run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_audit(path: Path) -> AuditRecord:
    return AuditRecord(path=path, payload=json.loads(path.read_text()))


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    value = float(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def best_runs_by_group(runs: list[RunRecord]) -> dict[tuple[str, str], RunRecord]:
    best: dict[tuple[str, str], RunRecord] = {}
    for run in runs:
        key = (run.benchmark, run.method)
        current = best.get(key)
        if current is None or run.accuracy > current.accuracy:
            best[key] = run
    return best


def write_accuracy_compute_plot(runs: list[RunRecord], out_path: Path) -> None:
    benchmarks = sorted({run.benchmark for run in runs})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4.5), squeeze=False)
    for axis, benchmark in zip(axes[0], benchmarks):
        benchmark_runs = [run for run in runs if run.benchmark == benchmark]
        for method in sorted({run.method for run in benchmark_runs}):
            method_runs = [run for run in benchmark_runs if run.method == method]
            xs = [run.compute for run in method_runs]
            ys = [run.accuracy for run in method_runs]
            axis.scatter(xs, ys, s=70, color=METHOD_COLORS.get(method, "#666666"), label=METHOD_LABELS.get(method, method))
            for run, x_val, y_val in zip(method_runs, xs, ys):
                axis.annotate(run.run_name.replace("hard_st_", "").replace("hybrid_es_", ""), (x_val, y_val), fontsize=7, xytext=(4, 4), textcoords="offset points")
        axis.set_title(BENCHMARK_LABELS.get(benchmark, benchmark))
        axis.set_xlabel("Mean Compute")
        axis.set_ylabel("Test Accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_route_behavior_plot(runs: list[RunRecord], benchmark: str, out_path: Path) -> None:
    benchmark_runs = [run for run in runs if run.benchmark == benchmark]
    if not benchmark_runs:
        return
    ordered = sorted(benchmark_runs, key=lambda run: (run.method, run.accuracy, run.run_name))
    labels = [run.run_name for run in ordered]
    accuracy = [run.accuracy for run in ordered]
    delay_rate = [run.delay_rate for run in ordered]
    early_exit = [run.early_exit_rate for run in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(ordered) * 0.6), 4.5), squeeze=False)
    for axis, values, title in [
        (axes[0, 0], accuracy, "Accuracy"),
        (axes[0, 1], delay_rate, "Delay Rate"),
        (axes[0, 2], early_exit, "Early Exit Rate"),
    ]:
        colors = [METHOD_COLORS.get(run.method, "#666666") for run in ordered]
        axis.bar(range(len(ordered)), values, color=colors)
        axis.set_xticks(range(len(ordered)), labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(f"{BENCHMARK_LABELS.get(benchmark, benchmark)} {title}")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_training_curves(runs: list[RunRecord], out_path: Path) -> None:
    selected = []
    best = best_runs_by_group(runs)
    for key in [
        ("long_horizon_memory_v1", "hard_st"),
        ("long_horizon_memory_v2", "hard_st"),
        ("long_horizon_memory_v2", "hybrid_es"),
    ]:
        if key in best:
            selected.append(best[key])
    if not selected:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), squeeze=False)
    for run in selected:
        rows = [row for row in load_metrics(run) if row.get("split") == "train"]
        if not rows:
            continue
        x_key = "generation" if run.method == "hybrid_es" else "step"
        xs = [row.get(x_key, 0) for row in rows]
        label = f"{run.method_label} {run.benchmark_label}"
        axes[0, 0].plot(xs, [row.get("accuracy", float("nan")) for row in rows], label=label, color=METHOD_COLORS.get(run.method, "#666666"))
        axes[1, 0].plot(xs, [row.get("delay_rate", float("nan")) for row in rows], label=label, color=METHOD_COLORS.get(run.method, "#666666"))

    axes[0, 0].set_title("Phase 2 Training Accuracy")
    axes[0, 0].set_ylabel("Train Accuracy")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_title("Phase 2 Training Delay Rate")
    axes[1, 0].set_ylabel("Delay Rate")
    axes[1, 0].set_xlabel("Step / Generation")
    axes[1, 0].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_es_diagnostics_plot(runs: list[RunRecord], out_path: Path) -> None:
    es_runs = [run for run in runs if run.method == "hybrid_es"]
    if not es_runs:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), squeeze=False)
    for run in es_runs:
        rows = [row for row in load_metrics(run) if row.get("split") == "train" and "generation" in row]
        if not rows:
            continue
        xs = [row["generation"] for row in rows]
        label = f"{run.run_name}"
        axes[0, 0].plot(xs, [row.get("reward", float("nan")) for row in rows], label=label)
        axes[1, 0].plot(xs, [row.get("reward_std", float("nan")) for row in rows], label=label)
    axes[0, 0].set_title("Hybrid ES Reward")
    axes[0, 0].set_ylabel("Population Mean Reward")
    axes[0, 0].grid(alpha=0.25)
    axes[1, 0].set_title("Hybrid ES Reward Std")
    axes[1, 0].set_ylabel("Reward Std")
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_table_rows(runs: list[RunRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in sorted(runs, key=lambda item: (item.benchmark, item.method, item.run_name)):
        rows.append(
            [
                run.run_name,
                run.benchmark_label,
                run.method_label,
                run.family_label,
                format_float(run.accuracy),
                format_float(run.delay_rate),
                format_float(run.route_match),
                format_float(run.early_exit_rate),
                format_float(run.compute),
                format_float(run.peak_train_memory_mb, 1),
                format_float(run.wall_time_sec, 1),
            ]
        )
    return rows


def audit_rows(audits: list[AuditRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for audit in audits:
        payload = audit.payload
        rows.append(
            [
                audit.benchmark_label,
                format_float(payload.get("heuristic_full_decode_accuracy")),
                format_float(payload.get("early_only_accuracy")),
                format_float(payload.get("final_only_accuracy")),
                str(payload.get("max_unique_oracle_route_patterns_per_batch", "-")),
                format_float(payload.get("mean_oracle_delays")),
                format_float(payload.get("mean_delay_penalty")),
                format_float(payload.get("break_even_ce_for_delay_vs_immediate_exit")),
                str(payload.get("delay_is_objectively_plausible", "-")),
            ]
        )
    return rows


def per_mode_rows(runs: list[RunRecord], benchmark: str) -> list[list[str]]:
    rows: list[list[str]] = []
    best = best_runs_by_group([run for run in runs if run.benchmark == benchmark])
    for method in ["hard_st", "hybrid_es", "soft"]:
        run = best.get((benchmark, method))
        if run is None:
            continue
        per_mode = run.test.get("per_mode") or {}
        for mode_name, mode_metrics in sorted(per_mode.items()):
            rows.append(
                [
                    run.method_label,
                    mode_name,
                    format_float(mode_metrics.get("accuracy")),
                    format_float(mode_metrics.get("delay_rate")),
                    format_float(mode_metrics.get("route_match")),
                    format_float(mode_metrics.get("early_exit_rate")),
                    format_float(mode_metrics.get("compute")),
                ]
            )
    return rows


def write_report(
    runs: list[RunRecord],
    audits: list[AuditRecord],
    out_path: Path,
) -> None:
    asset_dir = out_path.parent / "phase2_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    accuracy_plot = asset_dir / "accuracy_vs_compute.png"
    v1_plot = asset_dir / "benchmark_b_v1_route_behavior.png"
    v2_plot = asset_dir / "benchmark_b_v2_route_behavior.png"
    curves_plot = asset_dir / "training_curves.png"
    es_plot = asset_dir / "es_diagnostics.png"

    write_accuracy_compute_plot(runs, accuracy_plot)
    write_route_behavior_plot(runs, "long_horizon_memory_v1", v1_plot)
    write_route_behavior_plot(runs, "long_horizon_memory_v2", v2_plot)
    write_training_curves(runs, curves_plot)
    write_es_diagnostics_plot(runs, es_plot)

    best = best_runs_by_group(runs)
    best_v1_hard = best.get(("long_horizon_memory_v1", "hard_st"))
    best_v2_hard = best.get(("long_horizon_memory_v2", "hard_st"))
    best_v2_es = best.get(("long_horizon_memory_v2", "hybrid_es"))

    lines = [
        "# Phase 2 Report",
        "",
        "## Scope",
        "",
        "Second-round investigation of the long-horizon hard-routing packet GNN with an emphasis on Benchmark B collapse, route diagnostics, and EGGROLL-inspired hybrid ES follow-ups.",
        "",
        "## Benchmark Audit",
        "",
        markdown_table(
            [
                "Benchmark",
                "Heuristic Decode",
                "Early-Only",
                "Final-Only",
                "Unique Oracle Routes",
                "Mean Oracle Delays",
                "Mean Delay Penalty",
                "Break-Even CE",
                "Delay Plausible",
            ],
            audit_rows(audits),
        ),
        "",
        f"![Accuracy vs Compute]({accuracy_plot.relative_to(out_path.parent)})",
        "",
        f"![Benchmark B v1 Behavior]({v1_plot.relative_to(out_path.parent)})",
        "",
        f"![Benchmark B v2 Behavior]({v2_plot.relative_to(out_path.parent)})",
        "",
        f"![Training Curves]({curves_plot.relative_to(out_path.parent)})",
        "",
    ]

    if es_plot.exists():
        lines.extend(
            [
                f"![ES Diagnostics]({es_plot.relative_to(out_path.parent)})",
                "",
            ]
        )

    lines.extend(
        [
            "## Run Summary",
            "",
            markdown_table(
                [
                    "Run",
                    "Benchmark",
                    "Method",
                    "Family",
                    "Accuracy",
                    "Delay Rate",
                    "Route Match",
                    "Early Exit",
                    "Compute",
                    "Peak MB",
                    "Wall s",
                ],
                run_table_rows(runs),
            ),
            "",
            "## Best Available Results",
            "",
            markdown_table(
                ["Case", "Run", "Accuracy", "Delay Rate", "Route Match", "Early Exit", "Compute"],
                [
                    [
                        "Best Hard ST v1",
                        best_v1_hard.run_name if best_v1_hard else "-",
                        format_float(best_v1_hard.accuracy if best_v1_hard else None),
                        format_float(best_v1_hard.delay_rate if best_v1_hard else None),
                        format_float(best_v1_hard.route_match if best_v1_hard else None),
                        format_float(best_v1_hard.early_exit_rate if best_v1_hard else None),
                        format_float(best_v1_hard.compute if best_v1_hard else None),
                    ],
                    [
                        "Best Hard ST v2",
                        best_v2_hard.run_name if best_v2_hard else "-",
                        format_float(best_v2_hard.accuracy if best_v2_hard else None),
                        format_float(best_v2_hard.delay_rate if best_v2_hard else None),
                        format_float(best_v2_hard.route_match if best_v2_hard else None),
                        format_float(best_v2_hard.early_exit_rate if best_v2_hard else None),
                        format_float(best_v2_hard.compute if best_v2_hard else None),
                    ],
                    [
                        "Best Hybrid ES v2",
                        best_v2_es.run_name if best_v2_es else "-",
                        format_float(best_v2_es.accuracy if best_v2_es else None),
                        format_float(best_v2_es.delay_rate if best_v2_es else None),
                        format_float(best_v2_es.route_match if best_v2_es else None),
                        format_float(best_v2_es.early_exit_rate if best_v2_es else None),
                        format_float(best_v2_es.compute if best_v2_es else None),
                    ],
                ],
            ),
            "",
            "## Per-Mode Breakdown for Benchmark B v2",
            "",
            markdown_table(
                ["Method", "Mode", "Accuracy", "Delay Rate", "Route Match", "Early Exit", "Compute"],
                per_mode_rows(runs, "long_horizon_memory_v2"),
            ),
            "",
            "## Preliminary Phase 2 Read",
            "",
            "- Benchmark B v1 remains a true delayed-memory task, but the route is fixed rather than adaptive; this makes it a good memory stress test and a poor pure routing benchmark.",
            "- Benchmark B v2 is the better adaptive-routing benchmark because it mixes easy early exit with cases that should delay to a trigger or to the final query.",
            "- The main failure mode so far is not just early-exit collapse. Under oracle-routed v1 runs, the model can be made to delay almost perfectly while still staying at chance accuracy, which points to a content-memory bottleneck.",
            "- The adaptive delay-preservation gate creates a second failure mode: it can learn to preserve state almost perfectly, but then never writes the trigger payload. That is why phase 2 adds explicit delay-write diagnostics and a write-supervised intervention.",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dirs = [Path(item) for item in args.results_dir]
    runs = [load_run(run_dir) for run_dir in find_run_dirs(results_dirs)]
    audits = [load_audit(path) for path in find_audits(results_dirs)]
    write_report(runs, audits, Path(args.out))


if __name__ == "__main__":
    main()
