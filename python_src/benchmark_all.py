#!/usr/bin/env python3
"""
Consolidated benchmark: Run PyTorch once, then benchmark all C kernel versions.

This script:
1. Benchmarks PyTorch (naive + SDPA) once as baseline
2. Runs each C kernel binary, validates output, and computes speedups
3. Outputs results as text table (default) or JSON
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_artifacts, parse_c_time, run_c_binary


def parse_args():
    p = argparse.ArgumentParser(
        description="Consolidated benchmark: PyTorch baseline + all C kernel versions"
    )
    p.add_argument(
        "--bins",
        type=str,
        nargs="+",
        required=True,
        help="Paths to C kernel binaries to benchmark",
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--threads", type=int, default=1, help="Thread count for both C and PyTorch"
    )
    p.add_argument(
        "--atol", type=float, default=1e-4, help="Absolute tolerance for correctness"
    )
    p.add_argument(
        "--rtol", type=float, default=1e-4, help="Relative tolerance for correctness"
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of text table",
    )
    p.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save JSON results (implies --json)",
    )
    p.add_argument(
        "--use-srun",
        action="store_true",
        help="Use srun to launch binaries (for SLURM environments with proper CPU binding)",
    )
    return p.parse_args()


def extract_version(bin_path: str) -> str:
    """Extract version name from binary path (e.g., './cmhsa_v1.out' -> 'v1')."""
    name = Path(bin_path).stem  # 'cmhsa_v1'
    if name.startswith("cmhsa_"):
        return name[6:]  # 'v1'
    return name


@torch.no_grad()
def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Naive scaled dot-product attention with causal mask."""
    scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = Q.shape[-2]
    mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


@torch.no_grad()
def bench_naive_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    """
    Benchmark naive attention implementation.
    Returns (output tensor, per-iteration time in seconds).
    """
    out = None

    # Warmup
    for _ in range(warmup):
        naive_attention(Q, K, V)

    # Timed iterations
    t0 = time.perf_counter()
    for _ in range(iters):
        out = naive_attention(Q, K, V)
    t1 = time.perf_counter()

    assert out is not None
    return out, (t1 - t0) / iters


@torch.no_grad()
def bench_sdpa_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    """
    Benchmark PyTorch scaled_dot_product_attention.
    Returns (output tensor, per-iteration time in seconds).
    """
    out = None

    # Warmup
    for _ in range(warmup):
        F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )

    # Timed iterations
    t0 = time.perf_counter()
    for _ in range(iters):
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )
    t1 = time.perf_counter()

    assert out is not None
    return out, (t1 - t0) / iters


def print_text_table(
    config: dict,
    naive_per_iter: float,
    sdpa_per_iter: float,
    results: list[dict],
) -> None:
    """Print results as a formatted text table."""
    print(
        f"\n=== PyTorch Baseline (B={config['batch']} H={config['n_heads']} "
        f"S={config['seq_len']} D={config['head_dim']}, threads={config['threads']}) ==="
    )
    print(f"Naive PyTorch: {naive_per_iter:.6f} s/iter")
    print(f"Torch SDPA:    {sdpa_per_iter:.6f} s/iter")

    print("\n=== C Kernel Results ===")
    print(f"{'Version':<10} {'Time (s)':<12} {'vs Naive':<12} {'vs SDPA':<12}")
    print(f"{'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12}")

    for r in results:
        vs_naive = f"{r['speedup_vs_naive']:.2f}x"
        vs_sdpa = f"{r['speedup_vs_sdpa']:.2f}x"
        print(
            f"{r['version']:<10} {r['c_per_iter']:<12.6f} {vs_naive:<12} {vs_sdpa:<12}"
        )
    print()


def build_json_output(
    config: dict,
    naive_per_iter: float,
    sdpa_per_iter: float,
    results: list[dict],
) -> dict:
    """Build JSON output dictionary."""
    return {
        "threads": config["threads"],
        "config": config,
        "pytorch_baseline": {
            "naive_per_iter": naive_per_iter,
            "sdpa_per_iter": sdpa_per_iter,
        },
        "results": results,
    }


def print_json(
    config: dict,
    naive_per_iter: float,
    sdpa_per_iter: float,
    results: list[dict],
) -> None:
    """Print results as JSON to stdout."""
    output = build_json_output(config, naive_per_iter, sdpa_per_iter, results)
    print(json.dumps(output, indent=2))


def save_json(
    config: dict,
    naive_per_iter: float,
    sdpa_per_iter: float,
    results: list[dict],
    output_file: str,
) -> None:
    """Save results as JSON to a file."""
    output = build_json_output(config, naive_per_iter, sdpa_per_iter, results)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_file}")


def main():
    args = parse_args()

    # Strict thread control: no hidden parallelism for PyTorch
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)

    config = {
        "batch": args.batch,
        "n_heads": args.n_heads,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "threads": args.threads,
    }

    # Run the first C binary to get Q/K/V artifacts (C and Python RNGs differ,
    # so we use C's generated tensors for fair comparison)
    first_bin = args.bins[0]
    first_version = extract_version(first_bin)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        # Run first C binary to generate Q/K/V artifacts
        c_output = run_c_binary(
            first_bin,
            args.batch,
            args.n_heads,
            args.seq_len,
            args.head_dim,
            args.seed,
            args.threads,
            warmup=args.warmup,
            iters=args.iters,
            validate_outdir=outdir,
            use_srun=args.use_srun,
        )
        first_c_per_iter = parse_c_time(c_output)

        # Load Q/K/V from C binary artifacts for PyTorch benchmarks
        _, Q, K, V, first_out_c = load_artifacts(outdir)

    # Benchmark PyTorch once using C's Q/K/V
    out_naive, naive_per_iter = bench_naive_torch(Q, K, V, args.warmup, args.iters)
    out_sdpa, sdpa_per_iter = bench_sdpa_torch(Q, K, V, args.warmup, args.iters)

    # Use SDPA output as reference for validation
    out_ref = out_sdpa

    # Validate first C kernel
    assert torch.allclose(first_out_c, out_ref, rtol=args.rtol, atol=args.atol), (
        f"Validation failed for {first_version}: "
        f"max_abs_err={(first_out_c - out_ref).abs().max().item():.6g}"
    )

    # Store first result
    results = [
        {
            "version": first_version,
            "binary": first_bin,
            "c_per_iter": first_c_per_iter,
            "speedup_vs_naive": naive_per_iter / first_c_per_iter,
            "speedup_vs_sdpa": sdpa_per_iter / first_c_per_iter,
        }
    ]

    # Benchmark remaining C kernels
    for bin_path in args.bins[1:]:
        version = extract_version(bin_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            # Run C binary to get timing and artifacts
            c_output = run_c_binary(
                bin_path,
                args.batch,
                args.n_heads,
                args.seq_len,
                args.head_dim,
                args.seed,
                args.threads,
                warmup=args.warmup,
                iters=args.iters,
                validate_outdir=outdir,
                use_srun=args.use_srun,
            )
            c_per_iter = parse_c_time(c_output)

            # Load C output for validation
            _, _, _, _, out_c = load_artifacts(outdir)

            # Validate correctness against PyTorch reference
            assert torch.allclose(out_c, out_ref, rtol=args.rtol, atol=args.atol), (
                f"Validation failed for {version}: "
                f"max_abs_err={(out_c - out_ref).abs().max().item():.6g}"
            )

            results.append(
                {
                    "version": version,
                    "binary": bin_path,
                    "c_per_iter": c_per_iter,
                    "speedup_vs_naive": naive_per_iter / c_per_iter,
                    "speedup_vs_sdpa": sdpa_per_iter / c_per_iter,
                }
            )

    # Output results
    if args.output_file:
        # --output-file implies JSON output to file
        save_json(config, naive_per_iter, sdpa_per_iter, results, args.output_file)
    elif args.json:
        print_json(config, naive_per_iter, sdpa_per_iter, results)
    else:
        print_text_table(config, naive_per_iter, sdpa_per_iter, results)


if __name__ == "__main__":
    main()
