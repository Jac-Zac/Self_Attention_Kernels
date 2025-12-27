#!/usr/bin/env python3
"""
Self-Attention Kernel Scaling Analysis Plotter

Generates scalability plots for multi-threaded self-attention kernel implementations.
This script creates:
- Strong scaling speedup plots (log-log scale)
- Strong scaling efficiency plots
- Time vs threads plots

Input: Combined JSON file from merge_results.py containing benchmark data
       for multiple thread counts.

Usage:
    python scaling_plot.py --input results/combined.json --save-dir results/plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


# ---------------- Utility ----------------
def annotate_points(ax, x, y, flip_threshold=90, fontsize=8):
    """Add text annotations near each plotted point with smart placement."""
    for xi, yi in zip(x, y):
        if yi > flip_threshold:
            offset = (0, -10)
            va = "top"
        else:
            offset = (0, 5)
            va = "bottom"
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def load_combined_json(file_path: str) -> dict:
    """Load combined benchmark JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def extract_scaling_data(data: dict) -> dict:
    """
    Extract scaling data from combined JSON.

    Returns a dict with structure:
    {
        "threads": [1, 2, 4, 8, ...],
        "pytorch_naive": [t1, t2, ...],
        "pytorch_sdpa": [t1, t2, ...],
        "versions": {
            "v0": [t1, t2, ...],
            "v1": [t1, t2, ...],
            ...
        },
        "config": {...}  # from first run
    }
    """
    runs = data["runs"]
    threads = []
    pytorch_naive = []
    pytorch_sdpa = []
    versions: dict[str, list[float]] = {}

    for run in runs:
        threads.append(run["threads"])
        pytorch_naive.append(run["pytorch_baseline"]["naive_per_iter"])
        pytorch_sdpa.append(run["pytorch_baseline"]["sdpa_per_iter"])

        for result in run["results"]:
            version = result["version"]
            if version not in versions:
                versions[version] = []
            versions[version].append(result["c_per_iter"])

    return {
        "threads": np.array(threads),
        "pytorch_naive": np.array(pytorch_naive),
        "pytorch_sdpa": np.array(pytorch_sdpa),
        "versions": {k: np.array(v) for k, v in versions.items()},
        "config": runs[0]["config"] if runs else {},
    }


# ---------------- Strong Scaling Speedup ----------------
def plot_strong_scaling_speedup(
    scaling_data: dict, save_dir: str | None, show: bool
) -> None:
    """
    Plot strong scaling speedup (log-log scale).

    Speedup is calculated relative to each implementation's 1-thread performance.
    """
    threads = scaling_data["threads"]
    versions = scaling_data["versions"]
    pytorch_naive = scaling_data["pytorch_naive"]
    pytorch_sdpa = scaling_data["pytorch_sdpa"]
    config = scaling_data["config"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Ideal speedup line
    ideal_speedup = threads / threads[0]
    ax.plot(
        threads,
        ideal_speedup,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Ideal Speedup",
    )

    # Plot each C kernel version
    colors = plt.cm.tab10.colors
    for i, (version, times) in enumerate(sorted(versions.items())):
        baseline_time = times[0]  # 1-thread time for this version
        speedup = baseline_time / times
        ax.plot(
            threads,
            speedup,
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
            label=f"C Kernel {version}",
        )

    # Plot PyTorch versions
    naive_baseline = pytorch_naive[0]
    naive_speedup = naive_baseline / pytorch_naive
    ax.plot(
        threads,
        naive_speedup,
        marker="s",
        linestyle="--",
        color="purple",
        linewidth=2,
        markersize=6,
        label="PyTorch Naive",
    )

    sdpa_baseline = pytorch_sdpa[0]
    sdpa_speedup = sdpa_baseline / pytorch_sdpa
    ax.plot(
        threads,
        sdpa_speedup,
        marker="^",
        linestyle="--",
        color="green",
        linewidth=2,
        markersize=6,
        label="PyTorch SDPA",
    )

    # Formatting
    title = "Strong Scaling Speedup"
    if config:
        title += f" (B={config.get('batch')}, H={config.get('n_heads')}, S={config.get('seq_len')}, D={config.get('head_dim')})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(threads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "strong_scaling_speedup.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Strong Scaling Efficiency ----------------
def plot_strong_scaling_efficiency(
    scaling_data: dict, save_dir: str | None, show: bool
) -> None:
    """
    Plot strong scaling efficiency.

    Efficiency = (Speedup / Threads) * 100%
    """
    threads = scaling_data["threads"]
    versions = scaling_data["versions"]
    pytorch_naive = scaling_data["pytorch_naive"]
    pytorch_sdpa = scaling_data["pytorch_sdpa"]
    config = scaling_data["config"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # 100% ideal efficiency line
    ax.axhline(100, linestyle="--", color="black", linewidth=2, label="Ideal (100%)")

    # Plot each C kernel version
    colors = plt.cm.tab10.colors
    for i, (version, times) in enumerate(sorted(versions.items())):
        baseline_time = times[0]
        speedup = baseline_time / times
        ideal_speedup = threads / threads[0]
        efficiency = (speedup / ideal_speedup) * 100
        ax.plot(
            threads,
            efficiency,
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
            label=f"C Kernel {version}",
        )
        annotate_points(ax, threads, efficiency)

    # Plot PyTorch versions
    naive_baseline = pytorch_naive[0]
    naive_speedup = naive_baseline / pytorch_naive
    naive_ideal_speedup = threads / threads[0]
    naive_efficiency = (naive_speedup / naive_ideal_speedup) * 100
    ax.plot(
        threads,
        naive_efficiency,
        marker="s",
        linestyle="--",
        color="purple",
        linewidth=2,
        markersize=6,
        label="PyTorch Naive",
    )

    sdpa_baseline = pytorch_sdpa[0]
    sdpa_speedup = sdpa_baseline / pytorch_sdpa
    sdpa_ideal_speedup = threads / threads[0]
    sdpa_efficiency = (sdpa_speedup / sdpa_ideal_speedup) * 100
    ax.plot(
        threads,
        sdpa_efficiency,
        marker="^",
        linestyle="--",
        color="green",
        linewidth=2,
        markersize=6,
        label="PyTorch SDPA",
    )

    # Formatting
    title = "Strong Scaling Efficiency"
    if config:
        title += f" (B={config.get('batch')}, H={config.get('n_heads')}, S={config.get('seq_len')}, D={config.get('head_dim')})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("Efficiency (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="lower left")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "strong_scaling_efficiency.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Time vs Threads ----------------
def plot_time_vs_threads(scaling_data: dict, save_dir: str | None, show: bool) -> None:
    """
    Plot absolute execution time vs threads (log-log scale).
    """
    threads = scaling_data["threads"]
    versions = scaling_data["versions"]
    pytorch_naive = scaling_data["pytorch_naive"]
    pytorch_sdpa = scaling_data["pytorch_sdpa"]
    config = scaling_data["config"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each C kernel version
    colors = plt.cm.tab10.colors
    for i, (version, times) in enumerate(sorted(versions.items())):
        ax.plot(
            threads,
            times * 1000,  # Convert to ms
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
            label=f"C Kernel {version}",
        )

    # Plot PyTorch versions
    ax.plot(
        threads,
        pytorch_naive * 1000,
        marker="s",
        linestyle="--",
        color="purple",
        linewidth=2,
        markersize=6,
        label="PyTorch Naive",
    )

    ax.plot(
        threads,
        pytorch_sdpa * 1000,
        marker="^",
        linestyle="--",
        color="green",
        linewidth=2,
        markersize=6,
        label="PyTorch SDPA",
    )

    # Formatting
    title = "Execution Time vs Threads"
    if config:
        title += f" (B={config.get('batch')}, H={config.get('n_heads')}, S={config.get('seq_len')}, D={config.get('head_dim')})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("Time per Iteration (ms)", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(threads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "time_vs_threads.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Comparison Bar Chart ----------------
def plot_comparison_bar(scaling_data: dict, save_dir: str | None, show: bool) -> None:
    """
    Bar chart comparing all implementations at each thread count.
    """
    threads = scaling_data["threads"]
    versions = scaling_data["versions"]
    pytorch_naive = scaling_data["pytorch_naive"]
    pytorch_sdpa = scaling_data["pytorch_sdpa"]
    config = scaling_data["config"]

    # Build data for bar chart
    all_implementations = ["PyTorch Naive", "PyTorch SDPA"] + sorted(versions.keys())
    n_impl = len(all_implementations)
    n_threads = len(threads)

    x = np.arange(n_threads)
    total_width = 0.8
    bar_width = total_width / n_impl

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ["purple", "green"] + list(plt.cm.tab10.colors[: len(versions)])

    for i, impl in enumerate(all_implementations):
        if impl == "PyTorch Naive":
            times = pytorch_naive * 1000
        elif impl == "PyTorch SDPA":
            times = pytorch_sdpa * 1000
        else:
            times = versions[impl] * 1000

        offset = (i - n_impl / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            times,
            bar_width,
            label=impl,
            color=colors[i],
        )

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    # Formatting
    title = "Performance Comparison"
    if config:
        title += f" (B={config.get('batch')}, H={config.get('n_heads')}, S={config.get('seq_len')}, D={config.get('head_dim')})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("Time per Iteration (ms)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(threads)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "comparison_bar.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Self-Attention Kernel Scaling Analysis Plotter"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to combined benchmark JSON file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same directory as input file)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots to disk",
    )

    args = parser.parse_args()

    # Determine save directory
    save_dir = None
    if not args.no_save:
        if args.save_dir:
            save_dir = args.save_dir
        else:
            save_dir = str(Path(args.input).parent)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Plots will be saved to '{save_dir}/'")

    show_plots = not args.no_show

    # Load data
    print(f"Loading benchmark data from {args.input}...")
    data = load_combined_json(args.input)
    scaling_data = extract_scaling_data(data)

    print(
        f"Found {len(scaling_data['threads'])} thread configurations: {list(scaling_data['threads'])}"
    )
    print(
        f"Found {len(scaling_data['versions'])} kernel versions: {list(scaling_data['versions'].keys())}"
    )

    # Generate plots
    print("\nGenerating strong scaling speedup plot...")
    plot_strong_scaling_speedup(scaling_data, save_dir, show_plots)

    print("Generating strong scaling efficiency plot...")
    plot_strong_scaling_efficiency(scaling_data, save_dir, show_plots)

    print("Generating time vs threads plot...")
    plot_time_vs_threads(scaling_data, save_dir, show_plots)

    print("Generating comparison bar chart...")
    plot_comparison_bar(scaling_data, save_dir, show_plots)

    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
