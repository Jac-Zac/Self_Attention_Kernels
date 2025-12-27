"""Auto-detect and run appropriate plot based on data."""

import argparse
import sys
from pathlib import Path

from .utils import RESULTS_DIR, load_csv


def main():
    p = argparse.ArgumentParser(description="Auto-detect benchmark plot type")
    p.add_argument("-i", "--input", type=Path, default=RESULTS_DIR / "benchmark.csv")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("-t", "--threads", type=int, default=1)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    data = load_csv(args.input)
    thread_counts = set()
    for v in data.values():
        thread_counts.update(v.keys())

    print(
        f"Loaded {len(data)} versions, {len(thread_counts)} thread count(s): {sorted(thread_counts)}"
    )

    if len(thread_counts) == 1:
        from .single import plot

        output = args.output or RESULTS_DIR / "single_perf.png"
        plot(data, args.threads, output, not args.no_show)
    else:
        from .multi import plot

        output = args.output or RESULTS_DIR / "scaling.png"
        plot(data, output, not args.no_show)


if __name__ == "__main__":
    main()
