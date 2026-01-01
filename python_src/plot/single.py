"""Bar plot for single-threaded benchmark results."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import COLORS, RESULTS_DIR, load_csv, save_and_show, split_versions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-thread plot."""
    p = argparse.ArgumentParser(description="Single-thread benchmark bar plot")
    p.add_argument("-i", "--input", type=Path, default=RESULTS_DIR / "benchmark.csv")
    p.add_argument("-o", "--output", type=Path, default=RESULTS_DIR / "single_perf.png")
    p.add_argument("-t", "--threads", type=int, default=1)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def plot(data: dict, threads: int, output: Path | None, show: bool) -> None:
    """Bar chart: speedup vs PyTorch Naive and SDPA."""
    pytorch, kernels = split_versions(data)
    naive_t, sdpa_t = (
        pytorch["pytorch_naive"][threads],
        pytorch["pytorch_sdpa"][threads],
    )

    names = sorted(kernels.keys())
    times = [kernels[v][threads] for v in names]
    vs_naive = [naive_t / t for t in times]
    vs_sdpa = [sdpa_t / t for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    colors = [COLORS["kernels"][i % len(COLORS["kernels"])] for i in range(len(names))]

    for ax, speedups, baseline, title in [
        (ax1, vs_naive, ("naive", "PyTorch Naive"), "Speedup vs PyTorch Naive"),
        (ax2, vs_sdpa, ("sdpa", "PyTorch SDPA"), "Speedup vs PyTorch SDPA"),
    ]:
        bars = ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.5)
        for bar, s in zip(bars, speedups):
            ax.annotate(
                f"{s:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        ax.axhline(
            1.0,
            color=COLORS[baseline[0]],
            linestyle="--",
            linewidth=2,
            label=f"{baseline[1]} baseline",
        )
        ax.set_xlabel("Kernel")
        ax.set_ylabel("Speedup")
        ax.set_title(f"{title} (threads={threads})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C {v}" for v in names])
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Print summary
    print(f"\n{'Version':<12} {'Time (s)':<10} {'vs Naive':<10} {'vs SDPA':<10}")
    print("-" * 42)
    print(f"{'Naive':<12} {naive_t:<10.6f} {'1.00x':<10} {naive_t / sdpa_t:.2f}x")
    print(f"{'SDPA':<12} {sdpa_t:<10.6f} {sdpa_t / naive_t:.2f}x {'1.00x':<10}")
    for i, v in enumerate(names):
        print(
            f"{'C ' + v:<12} {times[i]:<10.6f} {vs_naive[i]:.2f}x{'':<5} {vs_sdpa[i]:.2f}x"
        )

    save_and_show(fig, output, show)


def main() -> None:
    """Entry point for single-thread benchmark plotting."""
    args = parse_args()
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    data = load_csv(args.input)
    print(f"Loaded {len(data)} versions from {args.input}")
    plot(data, args.threads, args.output, not args.no_show)


if __name__ == "__main__":
    main()
