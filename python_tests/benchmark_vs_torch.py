#!/usr/bin/env python3
"""
Fair CPU benchmark: C kernel vs PyTorch.

Key principles:
- Same Q/K/V values for both (loaded from C's artifacts)
- Identical thread counts (no hidden parallelism)
- Correctness assertion before reporting speedup
"""

import argparse
import re

# Allow imports when running as a script from the repo root
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_artifacts, run_c_binary


def parse_args():
    p = argparse.ArgumentParser(description="Fair CPU benchmark: C kernel vs PyTorch")
    p.add_argument(
        "--bin", type=str, default="./cmhsa.out", help="Path to built cmhsa binary"
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--head_dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument(
        "--threads", type=int, default=1, help="Thread count for both C and PyTorch"
    )
    p.add_argument(
        "--atol", type=float, default=1e-4, help="Absolute tolerance for correctness"
    )
    p.add_argument(
        "--rtol", type=float, default=1e-4, help="Relative tolerance for correctness"
    )
    return p.parse_args()


def parse_c_time(output: str) -> float:
    """Extract per-iteration time in seconds from C binary output."""
    m = re.search(r"CPU attention forward \(per-iter\):\s*([0-9.]+)\s*s", output)
    if not m:
        raise RuntimeError(
            "Could not parse per-iter time from binary output.\nOutput was:\n" + output
        )
    return float(m.group(1))


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
    Benchmark naive attention implementation on the given Q/K/V.
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
def bench_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    """
    Benchmark PyTorch scaled_dot_product_attention on the given Q/K/V.
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


def main():
    args = parse_args()

    # Strict thread control: no hidden parallelism for PyTorch
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        # Run C binary to generate artifacts and get timing
        c_output = run_c_binary(
            args.bin,
            args.batch,
            args.n_heads,
            args.seq_len,
            args.head_dim,
            args.seed,
            args.threads,
            warmup=args.warmup,
            iters=args.iters,
            validate_outdir=outdir,
        )
        c_per_iter = parse_c_time(c_output)

        # Load C's Q/K/V/out for fair comparison
        meta, Q, K, V, out_c = load_artifacts(outdir)

        # Benchmark naive PyTorch on the exact same inputs
        out_naive, naive_per_iter = bench_naive_torch(Q, K, V, args.warmup, args.iters)

        # Correctness assertion: C and naive PyTorch must produce the same output
        assert torch.allclose(out_naive, out_c, rtol=args.rtol, atol=args.atol), (
            f"Output mismatch between C and naive PyTorch! "
            f"max_abs_err={(out_naive - out_c).abs().max().item():.6g}"
        )

        # Benchmark optimized PyTorch on the exact same inputs
        out_torch, torch_per_iter = bench_torch(Q, K, V, args.warmup, args.iters)

        # Correctness assertion: C and PyTorch must produce the same output
        assert torch.allclose(out_torch, out_c, rtol=args.rtol, atol=args.atol), (
            f"Output mismatch between C and PyTorch! "
            f"max_abs_err={(out_torch - out_c).abs().max().item():.6g}"
        )

        # Report results
        print(
            f"Shapes: B={args.batch} H={args.n_heads} S={args.seq_len} D={args.head_dim}"
        )
        print(f"Threads: {args.threads}")
        print(f"C kernel      (per-iter): {c_per_iter:.6f} s")
        print(f"Naive PyTorch (per-iter): {naive_per_iter:.6f} s")
        print(f"Torch SDPA    (per-iter): {torch_per_iter:.6f} s")

        speedup_naive = naive_per_iter / c_per_iter
        speedup_sdpa = torch_per_iter / c_per_iter
        print(f"Speedup (Naive / C): {speedup_naive:.2f}x")
        print(f"Speedup (SDPA / C):  {speedup_sdpa:.2f}x")


if __name__ == "__main__":
    main()
