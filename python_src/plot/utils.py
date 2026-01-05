"""Shared plotting utilities and constants."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# Nord Theme Colors
COLORS = {
    "ideal": "#4C566A",
    "naive": "#B48EAD",
    "sdpa": "#A3BE8C",
    "kernels": [
        "#88C0D0",
        "#BF616A",
        "#D08770",
        "#EBCB8B",
        "#8FBCBB",
        "#81A1C1",
        "#5E81AC",
    ],
}


def load_csv(path: Path) -> dict[str, dict[int, float]]:
    """Load benchmark CSV into {version: {threads: time_s}}."""
    data: dict[str, dict[int, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ver, thr, time = row["version"], int(row["threads"]), float(row["time_s"])
            data.setdefault(ver, {})[thr] = time
    return data


def load_csv_with_backend(path: Path) -> tuple[dict[str, dict[int, float]], str | None]:
    """
    Load benchmark CSV into {version: {threads: time_s}} and extract backend.

    Returns:
        tuple: (data dict, backend string or None if not present)
    """
    data: dict[str, dict[int, float]] = {}
    backend: str | None = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ver, thr, time = row["version"], int(row["threads"]), float(row["time_s"])
            data.setdefault(ver, {})[thr] = time
            if backend is None and "backend" in row:
                backend = row["backend"]
    return data, backend


def split_versions(data: dict) -> tuple[dict, dict]:
    """Split into (pytorch_versions, kernel_versions)."""
    pytorch = {k: v for k, v in data.items() if k.startswith("pytorch")}
    kernels = {k: v for k, v in data.items() if not k.startswith("pytorch")}
    return pytorch, kernels


def save_and_show(fig, path: Path | None, show: bool) -> None:
    """Save figure to path and/or display it."""
    plt.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_speedup_bars(
    data: dict,
    threads: int,
    output: Path | None,
    show: bool,
    kernel_prefix: str = "C",
    title_suffix: str = "",
) -> None:
    """
    Shared bar chart for speedup comparisons vs PyTorch Naive and SDPA.

    Args:
        data: Benchmark data {version: {threads: time_s}}
        threads: Thread count to plot
        output: Output file path (or None to skip saving)
        show: Whether to display the plot
        kernel_prefix: Label prefix for kernels (e.g., "C" or "CUDA")
        title_suffix: Suffix for plot titles (e.g., "(threads=1)" or "(GPU)")
    """
    pytorch, kernels = split_versions(data)
    naive_t = pytorch["pytorch_naive"][threads]
    sdpa_t = pytorch["pytorch_sdpa"][threads]

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
        ax.set_title(f"{title} {title_suffix}".strip())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{kernel_prefix} {v}" for v in names])
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Print summary
    print(f"\n{'Version':<12} {'Time (s)':<10} {'vs Naive':<10} {'vs SDPA':<10}")
    print("-" * 42)
    print(f"{'Naive':<12} {naive_t:<10.6f} {'1.00x':<10} {naive_t / sdpa_t:.2f}x")
    print(f"{'SDPA':<12} {sdpa_t:<10.6f} {sdpa_t / naive_t:.2f}x {'1.00x':<10}")
    for i, v in enumerate(names):
        label = f"{kernel_prefix} {v}"
        print(
            f"{label:<12} {times[i]:<10.6f} {vs_naive[i]:.2f}x{'':<5} {vs_sdpa[i]:.2f}x"
        )

    save_and_show(fig, output, show)
