#!/usr/bin/env python3
"""Render validation-loss, timing, and cosine-similarity plots from logs.

The follow-up logs contain machine-readable ``BRANCHSTATS`` records.  The
original 3,072-wide run predates those records, so its mean similarity is read
from the human-readable ``W_a-W_b similarity`` lines.

Example:
    python plot_experiment_results.py \
      --baseline-log ../baseline_lambda.txt \
      --untied-log ../simple_product.txt \
      --followup-cos0-log ../latest/0.7.txt \
      --followup-cos0999-log ../latest/0.9.txt \
      --output-dir figures
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


BASELINE_COLOR = "#0072B2"  # Okabe-Ito blue
VARIANT_COLOR = "#D55E00"   # Okabe-Ito vermillion
FOLLOWUP_ZERO_COLOR = "#009E73"  # Okabe-Ito green


def set_publication_style() -> None:
    """Use a compact, colorblind-safe style suitable for papers and workshops."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


MEAN_RE = re.compile(r"W_a-W_b similarity:\s*([-+]?\d*\.\d+|\d+)")
STEP_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:")
VAL_LOSS_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([-+]?\d*\.\d+|\d+)\s+train_time:([-+]?\d*\.\d+|\d+)ms"
)
STATS_RE = re.compile(r"^BRANCHSTATS\s+(\{.*\})$", re.MULTILINE)


def read_original(path: Path) -> list[dict]:
    """Read the original run, which logs a mean similarity only."""
    records: list[dict] = []
    current_step: int | None = None
    for line in path.read_text().splitlines():
        match = STEP_RE.search(line)
        if match:
            current_step = int(match.group(1))
        match = MEAN_RE.search(line)
        if match and current_step is not None:
            records.append({"step": current_step, "mean_cosine": float(match.group(1))})
    if not records:
        raise ValueError(f"No similarity records found in {path}")
    return records


def read_followup(path: Path) -> list[dict]:
    """Read follow-up runs with a mean and one value per Transformer layer."""
    records = []
    for match in STATS_RE.finditer(path.read_text()):
        item = json.loads(match.group(1))
        item["mean_cosine"] = sum(item["cos_per_layer"]) / len(item["cos_per_layer"])
        records.append(item)
    if not records:
        raise ValueError(f"No BRANCHSTATS records found in {path}")
    return records


def read_validation_loss(path: Path) -> list[dict]:
    """Read validation losses, keeping only the final value for each step."""
    records: dict[int, dict] = {}
    for match in VAL_LOSS_RE.finditer(path.read_text()):
        step = int(match.group(1))
        records[step] = {
            "step": step,
            "val_loss": float(match.group(3)),
            "train_time_s": float(match.group(4)) / 1000.0,
        }
    if not records:
        raise ValueError(f"No validation-loss records found in {path}")
    return [records[step] for step in sorted(records)]


def write_csv(output_dir: Path, runs: dict[str, list[dict]]) -> None:
    rows = []
    for run, records in runs.items():
        for record in records:
            rows.append({
                "run": run,
                "step": record["step"],
                "mean_cosine": record["mean_cosine"],
                "init_align": record.get("init_align", ""),
                "hidden_dimension": record.get("hdim", ""),
            })
    with (output_dir / "cosine_similarity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_by_step_and_time(output_dir: Path, baseline: list[dict], untied: list[dict]) -> None:
    """Compare loss convergence by optimization step and by wall-clock time."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True, constrained_layout=True)

    def draw(ax, x_key: str) -> None:
        ax.plot(
            [r[x_key] for r in baseline],
            [r["val_loss"] for r in baseline],
            color=BASELINE_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            label="Squared ReLU",
            zorder=3,
        )
        ax.plot(
            [r[x_key] for r in untied],
            [r["val_loss"] for r in untied],
            color=VARIANT_COLOR,
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=4.5,
            label="Untied ReLU product",
            zorder=2,
        )
        ax.grid()
        ax.legend(frameon=False, loc="upper right")

    draw(axes[0], "step")
    axes[0].set_title("Validation loss convergence")
    axes[0].set(xlabel="Training step", ylabel="Validation loss", xlim=(-25, 1775))

    draw(axes[1], "train_time_s")
    axes[1].set_title("Validation loss vs. wall-clock time")
    axes[1].set(xlabel="Training time (s)", xlim=(-3, untied[-1]["train_time_s"] * 1.04))

    baseline_time = baseline[-1]["train_time_s"]
    untied_time = untied[-1]["train_time_s"]
    slowdown = 100.0 * (untied_time / baseline_time - 1.0)
    annotation_y = 4.35
    axes[1].vlines(
        [baseline_time, untied_time],
        ymin=axes[1].get_ylim()[0],
        ymax=annotation_y + 0.25,
        colors=[BASELINE_COLOR, VARIANT_COLOR],
        linestyles=":",
        linewidth=1.4,
    )
    axes[1].annotate(
        "",
        xy=(untied_time, annotation_y),
        xytext=(baseline_time, annotation_y),
        arrowprops={"arrowstyle": "<->", "color": VARIANT_COLOR, "lw": 1.5},
    )
    axes[1].text(
        (baseline_time + untied_time) / 2,
        annotation_y + 0.18,
        f"+{slowdown:.1f}%",
        color=VARIANT_COLOR,
        ha="center",
        va="bottom",
    )

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"loss_convergence_and_time.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_original_similarity(output_dir: Path, untied_cosine: list[dict]) -> None:
    """Plot mean branch similarity for the original untied run."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot([r["step"] for r in untied_cosine], [r["mean_cosine"] for r in untied_cosine], color=VARIANT_COLOR, linewidth=2.4)
    ax.set(xlabel="Training step", ylabel="Mean cosine similarity across layers", xlim=(0, 1750), ylim=(-0.03, 1.03))
    ax.grid(axis="y")
    final = untied_cosine[-1]
    ax.annotate(
        f"Final: {final['mean_cosine']:.4f}",
        xy=(final["step"], final["mean_cosine"]),
        xytext=(-92, 18),
        textcoords="offset points",
        color=VARIANT_COLOR,
        arrowprops={"arrowstyle": "-", "color": VARIANT_COLOR, "lw": 1},
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"original_cosine_similarity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_followup_similarity(output_dir: Path, near_zero: list[dict], near_one: list[dict]) -> None:
    """Plot the mean similarity trajectories from both follow-up initializations."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(
        [r["step"] for r in near_zero],
        [r["mean_cosine"] for r in near_zero],
        color=FOLLOWUP_ZERO_COLOR,
        linewidth=2.4,
        label="Initial cosine ≈ 0",
    )
    ax.plot(
        [r["step"] for r in near_one],
        [r["mean_cosine"] for r in near_one],
        color=VARIANT_COLOR,
        linewidth=2.4,
        label="Initial cosine ≈ 0.999",
    )
    ax.set(xlabel="Training step", ylabel="Mean cosine similarity across layers", xlim=(0, 1750), ylim=(-0.03, 1.03))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="center right")
    for records, color in (
        (near_zero, FOLLOWUP_ZERO_COLOR),
        (near_one, VARIANT_COLOR),
    ):
        final = records[-1]
        ax.annotate(
            f"Final: {final['mean_cosine']:.3f}",
            xy=(final["step"], final["mean_cosine"]),
            xytext=(-100, -5),
            textcoords="offset points",
            color=color,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1},
        )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"followup_cosine_similarity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_publication_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--untied-log", type=Path, required=True)
    parser.add_argument("--followup-cos0-log", type=Path, required=True)
    parser.add_argument("--followup-cos0999-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "Original run (initial cosine ≈ 0; d_ff = 3,072)": read_original(args.untied_log),
        "Follow-up (initial cosine ≈ 0; d_ff = 2,048)": read_followup(args.followup_cos0_log),
        "Follow-up (initial cosine ≈ 0.999; d_ff = 2,048)": read_followup(args.followup_cos0999_log),
    }
    write_csv(args.output_dir, runs)
    plot_loss_by_step_and_time(
        args.output_dir,
        read_validation_loss(args.baseline_log),
        read_validation_loss(args.untied_log),
    )
    plot_original_similarity(args.output_dir, runs["Original run (initial cosine ≈ 0; d_ff = 3,072)"])
    plot_followup_similarity(
        args.output_dir,
        runs["Follow-up (initial cosine ≈ 0; d_ff = 2,048)"],
        runs["Follow-up (initial cosine ≈ 0.999; d_ff = 2,048)"],
    )


if __name__ == "__main__":
    main()
