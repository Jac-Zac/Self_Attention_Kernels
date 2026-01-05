"""Auto-detect and run appropriate plot based on data."""

import argparse
import sys
from pathlib import Path

from .utils import RESULTS_DIR, load_csv_with_backend


def main() -> None:
    """Auto-detect benchmark type and run appropriate plot."""
    p = argparse.ArgumentParser(description="Auto-detect benchmark plot type")
    p.add_argument("-i", "--input", type=Path, default=RESULTS_DIR / "benchmark.csv")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("-t", "--threads", type=int, default=1)
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--backend",
        type=str,
        choices=["single", "multi", "cuda"],
        default=None,
        help="Force plot type (auto-detected from CSV if not specified)",
    )
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    data, csv_backend = load_csv_with_backend(args.input)
    thread_counts = set()
    for v in data.values():
        thread_counts.update(v.keys())

    # Determine backend: CLI flag > CSV column > auto-detect from thread counts
    backend = args.backend or csv_backend

    print(
        f"Loaded {len(data)} versions, {len(thread_counts)} thread count(s): {sorted(thread_counts)}"
    )
    if backend:
        print(f"Backend: {backend}")

    # Select plot type based on backend or thread count heuristic
    if backend == "cuda":
        from .cuda import plot

        output = args.output or RESULTS_DIR / "cuda_perf.png"
        plot(data, args.threads, output, not args.no_show)
    elif backend == "multi" or len(thread_counts) > 1:
        from .multi import plot

        output = args.output or RESULTS_DIR / "strong_scaling.png"
        plot(data, output, not args.no_show)
    else:
        # single-thread or unknown backend with single thread count
        from .single import plot

        output = args.output or RESULTS_DIR / "single_perf.png"
        plot(data, args.threads, output, not args.no_show)


if __name__ == "__main__":
    main()
