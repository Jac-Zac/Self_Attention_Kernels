#!/usr/bin/env python3
"""
Unified benchmark script for CMHSA kernels.

Outputs results as CSV for easy analysis and plotting.

Usage:
    # Single run at specific thread count
    python benchmark.py --bins ./cmhsa_v0.out ./cmhsa_v1.out --threads 8

    # Scaling analysis (multiple thread counts)
    python benchmark.py --bins ./cmhsa_v0.out --threads 1 2 4 8 16
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from utils import (
    RESULTS_DIR,
    load_artifacts,
    parse_c_time,
    parse_gpu_info,
    run_c_binary,
    tmp_artifacts_dir,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark script."""
    p = argparse.ArgumentParser(description="Benchmark CMHSA kernels (CSV output)")
    p.add_argument(
        "--bins", type=str, nargs="+", required=True, help="C kernel binaries"
    )
    p.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["single", "multi", "cuda"],
        help="Backend type: single-thread, multi-thread, or CUDA",
    )
    p.add_argument(
        "--threads", type=int, nargs="+", default=[1], help="Thread count(s)"
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--output", "-o", type=str, default=None, help="Output CSV path")
    p.add_argument("--use-srun", action="store_true", help="Use srun for SLURM")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run PyTorch on (default: cpu)",
    )
    return p.parse_args()


def extract_version(bin_path: str) -> str:
    """
    Extract version name from binary path.

    Args:
        bin_path: Path to binary (e.g., './cmhsa_v1.out')

    Returns:
        str: Version string (e.g., 'v1')
    """
    name = Path(bin_path).stem
    return name[6:] if name.startswith("cmhsa_") else name


@torch.no_grad()
def bench_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    warmup: int,
    iters: int,
    use_sdpa: bool,
) -> tuple[torch.Tensor, float]:
    """
    Benchmark PyTorch attention implementation.

    Args:
        Q: Query tensor [B, H, S, D]
        K: Key tensor [B, H, S, D]
        V: Value tensor [B, H, S, D]
        warmup: Number of warmup iterations
        iters: Number of timed iterations
        use_sdpa: Whether to use scaled_dot_product_attention (True)
                  or manual implementation (False)

    Returns:
        tuple: (output tensor, per-iteration time in seconds)
    """
    if use_sdpa:
        fn = lambda: F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )
    else:
        S = Q.shape[-2]
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1).to(Q.device)
        scale = Q.shape[-1] ** -0.5

        def fn():
            attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn = attn.masked_fill(mask, float("-inf"))
            return torch.matmul(F.softmax(attn, dim=-1), V)

    for _ in range(warmup):
        fn()
    out = fn()  # ensure at least one call
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    return out, (time.perf_counter() - t0) / iters


def main():
    args = parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            f"ERROR: --device cuda requested but CUDA is not available", file=sys.stderr
        )
        sys.exit(1)

    # Output path
    output_path = Path(args.output) if args.output else RESULTS_DIR / "benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_cuda = args.backend == "cuda"
    rows = []

    # Initialize GPU info for CUDA backends
    gpu_info = {}

    for threads in args.threads:
        # Run first binary to get Q/K/V artifacts and extract GPU info if needed
        first_bin = args.bins[0]
        with tmp_artifacts_dir() as outdir:
            c_output = run_c_binary(
                first_bin,
                args.batch,
                args.n_heads,
                args.seq_len,
                args.head_dim,
                args.seed,
                threads,
                args.warmup,
                args.iters,
                outdir,
                args.use_srun,
            )
            first_time = parse_c_time(c_output)
            _, Q, K, V, first_out_c = load_artifacts(outdir, device=args.device)

            # Extract GPU info for CUDA backend (only on first iteration)
            if is_cuda and not gpu_info:
                gpu_info = parse_gpu_info(c_output)

        # Print benchmark header with GPU info for CUDA
        if is_cuda:
            gpu_name = gpu_info.get("name", "Unknown GPU")
            print(f"\n=== Benchmarking with {threads} thread(s) [GPU: {gpu_name}] ===")
        else:
            print(f"\n=== Benchmarking with {threads} thread(s) ===")

        torch.set_num_threads(max(1, threads))
        torch.set_num_interop_threads(1)

        # Benchmark PyTorch
        _, naive_time = bench_torch(Q, K, V, args.warmup, args.iters, use_sdpa=False)
        out_ref, sdpa_time = bench_torch(
            Q, K, V, args.warmup, args.iters, use_sdpa=True
        )
        out_ref = out_ref.to(device)

        # Validate first kernel
        assert torch.allclose(
            first_out_c, out_ref, rtol=args.rtol, atol=args.atol
        ), f"Validation failed for {extract_version(first_bin)}"

        # Record PyTorch results
        rows.append(
            {"threads": threads, "version": "pytorch_naive", "time_s": naive_time}
        )
        rows.append(
            {"threads": threads, "version": "pytorch_sdpa", "time_s": sdpa_time}
        )
        rows.append(
            {
                "threads": threads,
                "version": extract_version(first_bin),
                "time_s": first_time,
            }
        )

        # Benchmark remaining kernels
        for bin_path in args.bins[1:]:
            version = extract_version(bin_path)
            with tmp_artifacts_dir() as outdir:
                c_output = run_c_binary(
                    bin_path,
                    args.batch,
                    args.n_heads,
                    args.seq_len,
                    args.head_dim,
                    args.seed,
                    threads,
                    args.warmup,
                    args.iters,
                    outdir,
                    args.use_srun,
                )
                c_time = parse_c_time(c_output)
                _, _, _, _, out_c = load_artifacts(outdir, device=args.device)

            assert torch.allclose(
                out_c, out_ref, rtol=args.rtol, atol=args.atol
            ), f"Validation failed for {version}"
            rows.append({"threads": threads, "version": version, "time_s": c_time})

        # Print summary for this thread count
        print(f"  pytorch_naive: {naive_time:.6f}s")
        print(f"  pytorch_sdpa:  {sdpa_time:.6f}s")
        for r in rows:
            if r["threads"] == threads and not r["version"].startswith("pytorch"):
                print(
                    f"  {r['version']}: {r['time_s']:.6f}s (vs sdpa: {sdpa_time / r['time_s']:.2f}x)"
                )

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threads", "version", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
