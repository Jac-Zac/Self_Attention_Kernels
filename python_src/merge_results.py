#!/usr/bin/env python3
"""
Merge multiple benchmark JSON files into a single combined JSON.

This script reads individual benchmark JSON files (one per thread count)
and combines them into a single file suitable for plotting.

Usage:
    python merge_results.py results/benchmark_*threads.json --output results/combined.json
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge multiple benchmark JSON files into one combined file"
    )
    p.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input JSON files to merge (e.g., results/benchmark_*threads.json)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the combined JSON file",
    )
    p.add_argument(
        "--delete-inputs",
        action="store_true",
        help="Delete input files after successful merge",
    )
    return p.parse_args()


def load_json(file_path: str) -> dict:
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    runs = []
    for input_file in args.input_files:
        path = Path(input_file)
        if not path.exists():
            print(f"Warning: File not found: {input_file}", file=sys.stderr)
            continue

        try:
            data = load_json(input_file)
            # Ensure threads field is present at top level
            if "threads" not in data:
                # Try to extract from config
                if "config" in data and "threads" in data["config"]:
                    data["threads"] = data["config"]["threads"]
                else:
                    print(
                        f"Warning: No threads info in {input_file}, skipping",
                        file=sys.stderr,
                    )
                    continue
            runs.append(data)
            print(f"Loaded: {input_file} (threads={data['threads']})")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {input_file}: {e}", file=sys.stderr)
            continue

    if not runs:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    # Sort runs by thread count
    runs.sort(key=lambda x: x["threads"])

    # Build combined output
    combined = {"runs": runs}

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined {len(runs)} benchmark runs into {args.output}")
    print(f"Thread counts: {[r['threads'] for r in runs]}")

    # Delete input files if requested
    if args.delete_inputs:
        for input_file in args.input_files:
            path = Path(input_file)
            if path.exists():
                path.unlink()
                print(f"Deleted: {input_file}")


if __name__ == "__main__":
    main()
