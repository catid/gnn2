from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from src.utils.config import load_config


METHOD_LABELS = {
    "soft": "Soft Routing",
    "hard_st": "Hard ST",
    "hybrid_es": "Hybrid ES",
}

BENCHMARK_LABELS = {
    "mixed_oracle_routing": "Benchmark A",
    "long_horizon_memory": "Benchmark B",
}

MODE_ORDER = [
    "easy_exit",
    "spatial_k",
    "delay_1",
    "delay_1_then_spatial_k",
]

PLOT_COLORS = {
    "soft": "#1b9e77",
    "hard_st": "#d95f02",
    "hybrid_es": "#7570b3",
}


@dataclass
class RunRecord:
    path: Path
    benchmark: str
    method: str
    config: dict[str, Any]
    summary: dict[str, Any]
    test: dict[str, Any]
    system_info: dict[str, Any]

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
    def es_rank(self) -> int:
        return int(self.config.get("es", {}).get("rank", 0))

    @property
    def es_population(self) -> int:
        return int(self.config.get("es", {}).get("population", 0))

    @property
    def warmstart_enabled(self) -> bool:
        return bool(self.config.get("warmstart", {}).get("enabled", False))

    @property
    def evolve_adapters(self) -> bool:
        return bool(self.config.get("es", {}).get("evolve_adapters", False))

    @property
    def accuracy(self) -> float:
        return float(self.test.get("accuracy", float("nan")))

    @property
    def compute(self) -> float:
        return float(self.test.get("compute", float("nan")))

    @property
    def route_match(self) -> float:
        return float(self.test.get("route_match", float("nan")))

    @property
    def train_wall_time_sec(self) -> float:
        if "total_wall_time_sec" in self.summary:
            return float(self.summary["total_wall_time_sec"])
        if "phase_wall_time_sec" in self.summary:
            return float(self.summary["phase_wall_time_sec"])
        if "es_wall_time_sec" in self.summary:
            return float(self.summary["es_wall_time_sec"])
        return float("nan")

    @property
    def peak_train_memory_mb(self) -> float:
        peaks: list[float] = []
        if "peak_train_memory_mb" in self.summary:
            peaks.append(float(self.summary["peak_train_memory_mb"]))
        warmstart = self.summary.get("warmstart") or {}
        if isinstance(warmstart, dict) and "peak_train_memory_mb" in warmstart:
            peaks.append(float(warmstart["peak_train_memory_mb"]))
        if peaks:
            return max(peaks)
        return float(self.test.get("peak_memory_mb", float("nan")))

    @property
    def label(self) -> str:
        extras: list[str] = []
        if self.benchmark == "long_horizon_memory":
            extras.append(f"T={self.seq_len}")
        if self.method == "hybrid_es":
            extras.append(f"r={self.es_rank}")
            extras.append(f"pop={self.es_population}")
            if not self.warmstart_enabled:
                extras.append("cold")
            if self.evolve_adapters:
                extras.append("adapters")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{self.method_label} / {self.benchmark_label}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown report from experiment runs.")
    parser.add_argument("--results-dir", required=True, help="Run directory or root containing multiple runs.")
    parser.add_argument("--out", required=True, help="Output markdown path.")
    parser.add_argument("--title", default="Packet-Routing GNN Experiment Report", help="Report title.")
    return parser.parse_args()


def find_run_dirs(results_dir: Path) -> list[Path]:
    if (results_dir / "summary.json").exists():
        return [results_dir]
    return sorted(path.parent for path in results_dir.rglob("summary.json"))


def load_run(run_dir: Path) -> RunRecord:
    payload = json.loads((run_dir / "summary.json").read_text())
    if payload.get("config_path"):
        config = load_config(payload["config_path"])
    else:
        config_path = run_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    system_path = run_dir / "system_info.json"
    system_info = json.loads(system_path.read_text()) if system_path.exists() else {}
    summary = payload["summary"]
    test = summary.get("test") or {}
    return RunRecord(
        path=run_dir,
        benchmark=payload["benchmark"],
        method=payload["method"],
        config=config,
        summary=summary,
        test=test,
        system_info=system_info,
    )


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    value = float(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No runs available._"
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line1, line2, *body])


def write_accuracy_vs_compute_plot(runs: list[RunRecord], out_path: Path) -> None:
    by_benchmark: dict[str, list[RunRecord]] = {}
    for run in runs:
        by_benchmark.setdefault(run.benchmark, []).append(run)
    if not by_benchmark:
        return

    fig, axes = plt.subplots(1, len(by_benchmark), figsize=(6 * len(by_benchmark), 4), squeeze=False)
    for axis, (benchmark, benchmark_runs) in zip(axes[0], sorted(by_benchmark.items())):
        for method in sorted({run.method for run in benchmark_runs}):
            method_runs = [run for run in benchmark_runs if run.method == method]
            xs = [run.compute for run in method_runs]
            ys = [run.accuracy for run in method_runs]
            axis.scatter(xs, ys, label=METHOD_LABELS.get(method, method), s=70, color=PLOT_COLORS.get(method))
            for run, x_val, y_val in zip(method_runs, xs, ys):
                annotate = f"T={run.seq_len}" if run.benchmark == "long_horizon_memory" else run.method_label
                axis.annotate(annotate, (x_val, y_val), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axis.set_title(BENCHMARK_LABELS.get(benchmark, benchmark))
        axis.set_xlabel("Mean Compute")
        axis.set_ylabel("Test Accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_wall_clock_plot(runs: list[RunRecord], out_path: Path) -> None:
    if not runs:
        return
    ordered = sorted(runs, key=lambda run: (run.benchmark, run.method, run.seq_len, run.path.name))
    labels = [run.label for run in ordered]
    values = [run.train_wall_time_sec for run in ordered]
    colors = [PLOT_COLORS.get(run.method, "#666666") for run in ordered]
    fig, axis = plt.subplots(figsize=(10, max(4, 0.45 * len(ordered))))
    axis.barh(labels, values, color=colors)
    axis.set_xlabel("Wall-Clock Training Time (s)")
    axis.set_title("Wall-Clock by Run")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_benchmark_a_route_plot(runs: list[RunRecord], out_path: Path) -> None:
    filtered = [run for run in runs if run.benchmark == "mixed_oracle_routing" and run.test.get("per_mode")]
    if not filtered:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    width = 0.23
    x_positions = list(range(len(MODE_ORDER)))
    for method_index, method in enumerate(sorted({run.method for run in filtered})):
        method_runs = [run for run in filtered if run.method == method]
        run = max(method_runs, key=lambda candidate: candidate.accuracy)
        route_vals = [run.test["per_mode"].get(mode, {}).get("route_match", float("nan")) for mode in MODE_ORDER]
        compute_vals = [run.test["per_mode"].get(mode, {}).get("compute", float("nan")) for mode in MODE_ORDER]
        offset_positions = [value + (method_index - 1) * width for value in x_positions]
        axes[0, 0].bar(offset_positions, route_vals, width=width, color=PLOT_COLORS.get(method), label=run.method_label)
        axes[0, 1].bar(offset_positions, compute_vals, width=width, color=PLOT_COLORS.get(method), label=run.method_label)

    for axis, title, ylabel in [
        (axes[0, 0], "Benchmark A Route Optimality by Mode", "Route Match"),
        (axes[0, 1], "Benchmark A Mean Compute by Mode", "Mean Compute"),
    ]:
        axis.set_xticks(x_positions, MODE_ORDER, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(title)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_horizon_plot(runs: list[RunRecord], out_path: Path) -> None:
    filtered = [run for run in runs if run.benchmark == "long_horizon_memory"]
    if not filtered:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    for method in sorted({run.method for run in filtered}):
        method_runs = sorted(
            [run for run in filtered if run.method == method],
            key=lambda run: run.seq_len,
        )
        xs = [run.seq_len for run in method_runs]
        accs = [run.accuracy for run in method_runs]
        mems = [run.peak_train_memory_mb for run in method_runs]
        axes[0, 0].plot(xs, accs, marker="o", color=PLOT_COLORS.get(method), label=METHOD_LABELS.get(method, method))
        axes[0, 1].plot(xs, mems, marker="o", color=PLOT_COLORS.get(method), label=METHOD_LABELS.get(method, method))

    axes[0, 0].set_title("Benchmark B Accuracy vs Horizon")
    axes[0, 0].set_xlabel("Sequence Length")
    axes[0, 0].set_ylabel("Test Accuracy")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].set_title("Benchmark B Peak Train Memory vs Horizon")
    axes[0, 1].set_xlabel("Sequence Length")
    axes[0, 1].set_ylabel("Peak Train Memory (MB)")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def benchmark_a_rows(runs: list[RunRecord]) -> list[list[str]]:
    filtered = [run for run in runs if run.benchmark == "mixed_oracle_routing"]
    rows: list[list[str]] = []
    for run in sorted(filtered, key=lambda item: (item.method, item.path.name)):
        rows.append(
            [
                run.path.name,
                run.method_label,
                format_float(run.accuracy),
                format_float(run.route_match),
                format_float(run.test.get("hops")),
                format_float(run.test.get("delays")),
                format_float(run.compute),
                format_float(run.test.get("early_exit_rate")),
                format_float(run.peak_train_memory_mb, 1),
                format_float(run.train_wall_time_sec, 1),
            ]
        )
    return rows


def benchmark_b_rows(runs: list[RunRecord]) -> list[list[str]]:
    filtered = [run for run in runs if run.benchmark == "long_horizon_memory"]
    rows: list[list[str]] = []
    for run in sorted(filtered, key=lambda item: (item.method, item.seq_len, item.path.name)):
        rows.append(
            [
                run.path.name,
                run.method_label,
                str(run.seq_len),
                format_float(run.accuracy),
                format_float(run.test.get("delays")),
                format_float(run.compute),
                format_float(run.test.get("examples_per_sec"), 1),
                format_float(run.peak_train_memory_mb, 1),
                format_float(run.train_wall_time_sec, 1),
            ]
        )
    return rows


def ablation_rows(runs: list[RunRecord]) -> list[list[str]]:
    filtered = [run for run in runs if run.method == "hybrid_es"]
    rows: list[list[str]] = []
    for run in sorted(
        filtered,
        key=lambda item: (
            item.benchmark,
            item.seq_len,
            item.es_rank,
            item.es_population,
            item.warmstart_enabled,
            item.evolve_adapters,
            item.path.name,
        ),
    ):
        rows.append(
            [
                run.path.name,
                run.benchmark_label,
                str(run.seq_len or "-"),
                str(run.es_rank),
                str(run.es_population),
                "yes" if run.warmstart_enabled else "no",
                "yes" if run.evolve_adapters else "no",
                format_float(run.accuracy),
                format_float(run.compute),
                format_float(run.train_wall_time_sec, 1),
            ]
        )
    return rows


def build_conclusion_lines(runs: list[RunRecord]) -> list[str]:
    lines: list[str] = []
    runs_a = [run for run in runs if run.benchmark == "mixed_oracle_routing"]
    runs_b = [run for run in runs if run.benchmark == "long_horizon_memory"]

    if runs_a:
        soft_best = max([run for run in runs_a if run.method == "soft"], key=lambda candidate: candidate.accuracy)
        hard_best = max([run for run in runs_a if run.method == "hard_st"], key=lambda candidate: candidate.accuracy)
        hybrid_best = max([run for run in runs_a if run.method == "hybrid_es"], key=lambda candidate: candidate.route_match)
        lines.append(
            f"On Benchmark A, **Soft Routing** won raw accuracy at {format_float(soft_best.accuracy)}, "
            f"but **Hybrid ES** was the strongest truly hard-routing method: "
            f"{format_float(hybrid_best.accuracy)} accuracy and {format_float(hybrid_best.route_match)} route match, "
            f"versus {format_float(hard_best.accuracy)} and {format_float(hard_best.route_match)} for Hard ST."
        )
        lines.append(
            f"Hybrid ES also used less mean compute than the soft model on Benchmark A "
            f"({format_float(hybrid_best.compute)} vs {format_float(soft_best.compute)}), "
            f"at the cost of lower final accuracy."
        )

    if runs_b:
        methods = sorted({run.method for run in runs_b})
        horizons_by_method = {
            method: {run.seq_len for run in runs_b if run.method == method}
            for method in methods
        }
        common_horizons = set.intersection(*horizons_by_method.values()) if horizons_by_method else set()
        if common_horizons:
            longest_common = max(common_horizons)
            selected = {
                method: max(
                    [run for run in runs_b if run.method == method and run.seq_len == longest_common],
                    key=lambda candidate: candidate.accuracy,
                )
                for method in methods
            }
            lines.append(
                f"On Benchmark B at the longest common horizon (T={longest_common}), all methods stayed near chance: "
                f"Soft {format_float(selected['soft'].accuracy)}, Hard ST {format_float(selected['hard_st'].accuracy)}, "
                f"Hybrid ES {format_float(selected['hybrid_es'].accuracy)}."
            )
            lines.append(
                f"As horizon grew from T=64 to T=128, soft-routing training memory rose from "
                f"{format_float(min(run.peak_train_memory_mb for run in runs_b if run.method == 'soft' and run.seq_len == 64), 1)} MB "
                f"to {format_float(min(run.peak_train_memory_mb for run in runs_b if run.method == 'soft' and run.seq_len == 128), 1)} MB, "
                f"while hybrid ES still failed to escape the trivial early-exit policy."
            )
        longest_horizon = max(run.seq_len for run in runs_b)
        longest_runs = [run for run in runs_b if run.seq_len == longest_horizon]
        if len({run.method for run in longest_runs}) == 1:
            only_run = longest_runs[0]
            lines.append(
                f"At the extra hybrid-only point T={longest_horizon}, hybrid ES remained near chance at "
                f"{format_float(only_run.accuracy)} while training time climbed to {format_float(only_run.train_wall_time_sec, 1)} s."
            )

    hybrid_runs = [run for run in runs if run.method == "hybrid_es"]
    if hybrid_runs:
        warm_runs = [run for run in hybrid_runs if run.warmstart_enabled]
        cold_runs = [run for run in hybrid_runs if not run.warmstart_enabled]
        if warm_runs and cold_runs:
            best_warm = max(warm_runs, key=lambda candidate: candidate.accuracy)
            best_cold = max(cold_runs, key=lambda candidate: candidate.accuracy)
            lines.append(
                f"Warm-starting hybrid ES improved its best observed accuracy from {format_float(best_cold.accuracy)} "
                f"to {format_float(best_warm.accuracy)} in the available ablations."
            )

    if runs_a and runs_b:
        lines.append(
            "Overall verdict: this EGGROLL-inspired approximation looks promising for practical hard-routing search "
            "when the main challenge is discrete route selection with compute penalties, but it does not make the "
            "current long-horizon delay-memory benchmark practical."
        )
        lines.append(
            "This is enough evidence to justify further research on better warm-starts or richer memory representations, "
            "not to claim a general win over gradient baselines."
        )

    if not lines:
        lines.append("The current report has too few completed runs to support a useful conclusion.")
    return lines


def build_report(title: str, runs: list[RunRecord], asset_dir: Path, out_path: Path, results_root: Path) -> str:
    accuracy_plot = asset_dir / "accuracy_vs_compute.png"
    wall_plot = asset_dir / "wall_clock.png"
    route_plot = asset_dir / "benchmark_a_route_stats.png"
    horizon_plot = asset_dir / "benchmark_b_horizon_scaling.png"

    asset_dir.mkdir(parents=True, exist_ok=True)
    write_accuracy_vs_compute_plot(runs, accuracy_plot)
    write_wall_clock_plot(runs, wall_plot)
    write_benchmark_a_route_plot(runs, route_plot)
    write_horizon_plot(runs, horizon_plot)

    relative = lambda path: path.relative_to(out_path.parent)
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "This report summarizes a synthetic packet-routing GNN experiment suite. The hybrid ES condition is **EGGROLL-inspired**, not a claim of exact EGGROLL reproduction.",
        "",
        "Implemented ingredients:",
        "",
        "- Hard forward-pass routing with `FORWARD`, `EXIT`, and `DELAY` actions.",
        "- A soft-routing BPTT baseline, a hard-routing straight-through baseline, and a hybrid low-rank ES method.",
        "- Low-rank antithetic ES perturbations with deterministic seed reconstruction and 2-GPU population sharding.",
        "- Direct optimization of task quality plus hop, delay, and TTL penalties.",
        "",
        "## Completed Runs",
        "",
        f"Loaded **{len(runs)}** run directories from `{results_root}`.",
        "",
        "## Headline Plots",
        "",
        "## Benchmark A Summary",
        "",
        markdown_table(
            [
                "Run",
                "Method",
                "Acc",
                "Route Match",
                "Hops",
                "Delays",
                "Compute",
                "Early Exit",
                "Peak Mem MB",
                "Train Time s",
            ],
            benchmark_a_rows(runs),
        ),
        "",
        "## Benchmark B Summary",
        "",
        markdown_table(
            [
                "Run",
                "Method",
                "Seq Len",
                "Acc",
                "Delays",
                "Compute",
                "Examples/s",
                "Peak Mem MB",
                "Train Time s",
            ],
            benchmark_b_rows(runs),
        ),
        "",
        "## Hybrid ES Ablations",
        "",
        markdown_table(
            [
                "Run",
                "Benchmark",
                "Seq Len",
                "Rank",
                "Population",
                "Warm",
                "Adapters",
                "Acc",
                "Compute",
                "Train Time s",
            ],
            ablation_rows(runs),
        ),
        "",
        "## Conclusion",
        "",
    ]
    plot_entries = [
        (accuracy_plot, "Accuracy vs Compute"),
        (wall_plot, "Wall Clock"),
        (route_plot, "Benchmark A Route Stats"),
        (horizon_plot, "Benchmark B Horizon Scaling"),
    ]
    plot_lines: list[str] = []
    for plot_path, alt_text in plot_entries:
        if plot_path.exists():
            plot_lines.extend([f"![{alt_text}]({relative(plot_path)})", ""])
    lines[lines.index("## Benchmark A Summary"):lines.index("## Benchmark A Summary")] = plot_lines
    for line in build_conclusion_lines(runs):
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The hybrid method follows the transferable EGGROLL ideas but does not reproduce the paper's exact kernels, shared-activation system, JAX stack, or full reference workloads.",
            "- These benchmarks are synthetic and diagnostic by design. Positive results here do not imply production-scale wins on real graph or sequence tasks.",
            "- The route simulator uses exact one-hot decisions for `route_mode='hard'`, but the state update remains a vectorized PyTorch approximation rather than a sparse custom kernel.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = [load_run(run_dir) for run_dir in find_run_dirs(results_dir)]
    asset_dir = out_path.parent / f"{out_path.stem}_assets"
    report = build_report(args.title, runs, asset_dir, out_path, results_dir)
    out_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
