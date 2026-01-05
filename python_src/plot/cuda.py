"""Bar plot for CUDA/GPU benchmark results."""

import argparse
import sys
from pathlib import Path

from .utils import RESULTS_DIR, load_csv, plot_speedup_bars


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CUDA plot."""
    p = argparse.ArgumentParser(description="CUDA/GPU benchmark bar plot")
    p.add_argument("-i", "--input", type=Path, default=RESULTS_DIR / "benchmark.csv")
    p.add_argument("-o", "--output", type=Path, default=RESULTS_DIR / "cuda_perf.png")
    p.add_argument("-t", "--threads", type=int, default=1)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def plot(data: dict, threads: int, output: Path | None, show: bool) -> None:
    """Bar chart: speedup vs PyTorch SDPA (GPU comparison)."""
    plot_speedup_bars(
        data,
        threads,
        output,
        show,
        kernel_prefix="CUDA",
        title_suffix="(GPU)",
    )


def main() -> None:
    """Entry point for CUDA benchmark plotting."""
    args = parse_args()
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    data = load_csv(args.input)
    print(f"Loaded {len(data)} versions from {args.input}")
    plot(data, args.threads, args.output, not args.no_show)


if __name__ == "__main__":
    main()
