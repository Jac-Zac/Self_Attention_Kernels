"""Scaling plot for multi-threaded benchmark results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from .utils import COLORS, RESULTS_DIR, load_csv, save_and_show, split_versions


def parse_args():
    p = argparse.ArgumentParser(description="Multi-thread scaling plot")
    p.add_argument("-i", "--input", type=Path, default=RESULTS_DIR / "benchmark.csv")
    p.add_argument("-o", "--output", type=Path, default=RESULTS_DIR / "scaling.png")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def plot(data: dict, output: Path | None, show: bool) -> None:
    """Scaling plot: self-relative, vs naive, vs SDPA."""
    pytorch, kernels = split_versions(data)
    threads = np.array(sorted(next(iter(data.values())).keys()))

    if len(threads) < 2:
        raise ValueError(
            "Need multiple thread counts. Use plot.single for single-thread."
        )

    naive_t = np.array([pytorch["pytorch_naive"][t] for t in threads])
    sdpa_t = np.array([pytorch["pytorch_sdpa"][t] for t in threads])
    names = sorted(kernels.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Self-relative scaling
    ax = axes[0]
    ax.plot(
        threads, threads / threads[0], "--", color=COLORS["ideal"], lw=2, label="Ideal"
    )
    for i, v in enumerate(names):
        t = np.array([kernels[v][th] for th in threads])
        ax.plot(
            threads,
            t[0] / t,
            "o-",
            color=COLORS["kernels"][i % len(COLORS["kernels"])],
            lw=2,
            ms=6,
            label=f"C {v}",
        )
    ax.plot(
        threads,
        naive_t[0] / naive_t,
        "s--",
        color=COLORS["naive"],
        lw=2,
        ms=6,
        label="Naive",
    )
    ax.plot(
        threads,
        sdpa_t[0] / sdpa_t,
        "^--",
        color=COLORS["sdpa"],
        lw=2,
        ms=6,
        label="SDPA",
    )
    ax.set_title("Strong Scaling (Self-Relative)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(threads)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", lw=0.5)

    # Plot 2 & 3: vs Naive and vs SDPA
    for ax, base_t, base_name, color_key in [
        (axes[1], naive_t, "Naive", "naive"),
        (axes[2], sdpa_t, "SDPA", "sdpa"),
    ]:
        ax.axhline(
            1.0,
            color=COLORS[color_key],
            linestyle="--",
            lw=2,
            label=f"{base_name} baseline",
        )
        for i, v in enumerate(names):
            t = np.array([kernels[v][th] for th in threads])
            ax.plot(
                threads,
                base_t / t,
                "o-",
                color=COLORS["kernels"][i % len(COLORS["kernels"])],
                lw=2,
                ms=6,
                label=f"C {v}",
            )
        # Add other pytorch for reference
        other_t, other_name, other_color = (
            (sdpa_t, "SDPA", "sdpa")
            if base_name == "Naive"
            else (naive_t, "Naive", "naive")
        )
        ax.plot(
            threads,
            base_t / other_t,
            "^--" if base_name == "Naive" else "s--",
            color=COLORS[other_color],
            lw=2,
            ms=6,
            label=f"PyTorch {other_name}",
        )
        ax.set_title(f"Speedup vs PyTorch {base_name}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup")
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", lw=0.5)

    # Print summary
    print(f"\n{'Version':<12}", end="")
    for t in threads:
        print(f" {t:>5}T", end="")
    print("\n" + "-" * (12 + 7 * len(threads)))

    for label, times in [("Naive", naive_t), ("SDPA", sdpa_t)] + [
        (f"C {v}", [kernels[v][t] for t in threads]) for v in names
    ]:
        print(f"{label:<12}", end="")
        for t in times:
            print(f" {t:>5.4f}", end="")
        print()

    print(f"\nSpeedup vs Naive:")
    for v in names:
        t = np.array([kernels[v][th] for th in threads])
        print(
            f"{'C ' + v:<12}"
            + "".join(f" {naive_t[i] / t[i]:>5.2f}x" for i in range(len(threads)))
        )

    print(f"\nSpeedup vs SDPA:")
    for v in names:
        t = np.array([kernels[v][th] for th in threads])
        print(
            f"{'C ' + v:<12}"
            + "".join(f" {sdpa_t[i] / t[i]:>5.2f}x" for i in range(len(threads)))
        )

    save_and_show(fig, output, show)


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    data = load_csv(args.input)
    print(f"Loaded {len(data)} versions from {args.input}")
    plot(data, args.output, not args.no_show)


if __name__ == "__main__":
    main()
