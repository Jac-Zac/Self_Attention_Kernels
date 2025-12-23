#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="CPU benchmark: cmhsa vs PyTorch")
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
        "--threads", type=int, default=1, help="PyTorch intraop thread count (CPU)"
    )
    return p.parse_args()


def run_c_binary(
    bin_path: str,
    B: int,
    H: int,
    S: int,
    D: int,
    seed: int,
    warmup: int,
    iters: int,
    threads: int,
) -> float:
    cmd = [
        bin_path,
        "--batch",
        str(B),
        "--n_heads",
        str(H),
        "--seq_len",
        str(S),
        "--head_dim",
        str(D),
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--threads",
        str(max(1, threads)),
    ]
    out = subprocess.check_output(cmd, text=True)
    m = re.search(r"CPU attention forward \(per-iter\):\s*([0-9.]+)\s*s", out)
    if not m:
        raise RuntimeError(
            "Could not parse per-iter time from binary output.\nOutput was:\n" + out
        )
    return float(m.group(1))


@torch.no_grad()
def naive_scaled_dot_product_attention(Q, K, V, is_causal=True):
    # Naive implementation: Q @ K^T / sqrt(d_k), causal mask, softmax, @ V
    scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        S = Q.shape[-2]
        mask = torch.triu(
            torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


@torch.no_grad()
def bench_naive_torch(
    B: int, H: int, S: int, D: int, seed: int, warmup: int, iters: int
) -> float:
    torch.manual_seed(seed)
    # Q, K, V match layout used by your validator and binary
    Q = torch.randn(B, H, S, D, dtype=torch.float32, device="cpu").contiguous()
    K = torch.randn(B, H, S, D, dtype=torch.float32, device="cpu").contiguous()
    V = torch.randn(B, H, S, D, dtype=torch.float32, device="cpu").contiguous()

    # Warm-up iterations (not timed)
    for _ in range(warmup):
        naive_scaled_dot_product_attention(Q, K, V, is_causal=True)

    t0 = time.perf_counter()
    for _ in range(iters):
        naive_scaled_dot_product_attention(Q, K, V, is_causal=True)
    t1 = time.perf_counter()

    return (t1 - t0) / float(iters)


@torch.no_grad()
def bench_optimized_torch(
    B: int, H: int, S: int, D: int, seed: int, warmup: int, iters: int
) -> float:
    torch.manual_seed(seed)
    # Q, K, V match layout used by your validator and binary
    Q = torch.randn(B, H, S, D, dtype=torch.float32, device="cpu").contiguous()
    K = torch.randn(B, H, S, D, dtype=torch.float32, device="cpu").contiguous()
    V = torch.randn(B, H, S, D, dtype=torch.float32, device="cpu").contiguous()

    # Warm-up iterations (not timed)
    for _ in range(warmup):
        F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )

    t0 = time.perf_counter()
    for _ in range(iters):
        F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )
    t1 = time.perf_counter()

    return (t1 - t0) / float(iters)


def main():
    args = parse_args()

    # For fairness, control PyTorch threads here. You should also set
    # shell env for BLAS backends before launching this script, e.g.:
    # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    # This script controls PyTorch intraop and interop threads only.
    torch.set_num_threads(max(1, args.threads))
    # A conservative interop setting; adjust if you prefer
    torch.set_num_interop_threads(max(1, args.threads // 2))

    # Run C++ binary once; it reports warmups/iters internally and prints per-iter seconds
    c_per_iter = run_c_binary(
        args.bin,
        args.batch,
        args.n_heads,
        args.seq_len,
        args.head_dim,
        args.seed,
        args.warmup,
        args.iters,
        args.threads,
    )

    # Run PyTorch timings with same shape
    naive_t_per_iter = bench_naive_torch(
        args.batch,
        args.n_heads,
        args.seq_len,
        args.head_dim,
        args.seed,
        args.warmup,
        args.iters,
    )
    optimized_t_per_iter = bench_optimized_torch(
        args.batch,
        args.n_heads,
        args.seq_len,
        args.head_dim,
        args.seed,
        args.warmup,
        args.iters,
    )

    print(
        f"\nShapes: B={args.batch} H={args.n_heads} S={args.seq_len} D={args.head_dim}"
    )
    print(f"Naive PyTorch     (per-iter): {naive_t_per_iter:.6f} s")
    print(f"Optimized PyTorch (per-iter): {optimized_t_per_iter:.6f} s")
    if c_per_iter > 0.0:
        print(f"Speedup (Naive Torch/C++): {naive_t_per_iter / c_per_iter:.3f}x")
        print(
            f"Speedup (Optimized Torch/C++): {optimized_t_per_iter / c_per_iter:.3f}x\n"
        )


if __name__ == "__main__":
    main()
