#!/usr/bin/env python3
"""
Validate CMHSA C kernel output against PyTorch reference.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_artifacts, run_c_binary, tmp_artifacts_dir


def parse_args():
    p = argparse.ArgumentParser(description="Validate CMHSA against PyTorch")
    p.add_argument("--bin", type=str, default="./cmhsa.out", help="Path to binary")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--use-srun", action="store_true", help="Use srun for SLURM")
    return p.parse_args()


def main():
    args = parse_args()

    with tmp_artifacts_dir() as outdir:
        run_c_binary(
            args.bin,
            args.batch,
            args.n_heads,
            args.seq_len,
            args.head_dim,
            args.seed,
            args.threads,
            args.iters,
            validate_outdir=outdir,
            use_srun=args.use_srun,
        )
        _, Q, K, V, out_c = load_artifacts(outdir)

    out_torch = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    max_abs = (out_c - out_torch).abs().max().item()
    ok = torch.allclose(out_c, out_torch, rtol=args.rtol, atol=args.atol)

    print(f"max_abs_err={max_abs:.6g} rtol={args.rtol} atol={args.atol}")
    if not ok:
        print("FAIL: C output differs from PyTorch reference")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
