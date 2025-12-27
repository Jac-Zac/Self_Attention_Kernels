#!/usr/bin/env python3
"""
Float32-only PyTorch validator for CMHSA.
- Runs cmhsa.out in validation mode to produce raw float32 files in a temporary directory
- Reads q.bin, k.bin, v.bin, out.bin, meta.json
- Computes causal scaled dot-product attention in PyTorch
- Compares outputs with tolerances and exits non-zero on mismatch
- Uses a TemporaryDirectory so no persistent artifacts are left
"""

import argparse

# Allow imports when running as a script from the repo root
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Import from parent directory (python_src/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_artifacts, run_c_binary


def parse_args():
    p = argparse.ArgumentParser(description="Validate CMHSA against PyTorch")
    p.add_argument(
        "--bin", type=str, default="./cmhsa.out", help="Path to built cmhsa binary"
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--use-srun",
        action="store_true",
        help="Use srun to launch binaries (for SLURM environments with proper CPU binding)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)

        # Run C binary to generate artifacts
        run_c_binary(
            args.bin,
            args.batch,
            args.n_heads,
            args.seq_len,
            args.head_dim,
            args.seed,
            args.threads,
            validate_outdir=outdir,
            use_srun=args.use_srun,
        )

        # Load artifacts
        meta, Q, K, V, out_c = load_artifacts(outdir)

        # Compute PyTorch causal attention (torch applies 1/sqrt(D) internally)
        out_torch = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # Compare
        out_c_np = out_c.numpy()
        out_torch_np = out_torch.detach().cpu().numpy()

        diff = np.abs(out_c_np - out_torch_np)
        rel = diff / (np.maximum(np.abs(out_torch_np), 1e-12))
        max_abs = float(diff.max())
        max_rel = float(rel.max())
        ok = np.allclose(out_c_np, out_torch_np, rtol=args.rtol, atol=args.atol)

        print(
            f"Validation: max_abs={max_abs:.6g} max_rel={max_rel:.6g} rtol={args.rtol} atol={args.atol}"
        )
        if not ok:
            print("Mismatch: C output deviates from PyTorch reference")
            raise SystemExit(1)

        print("Validation PASS; no persistent artifacts (temp dir cleaned)")


if __name__ == "__main__":
    main()
