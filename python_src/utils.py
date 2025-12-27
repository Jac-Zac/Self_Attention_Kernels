#!/usr/bin/env python3
"""
Shared utilities for CMHSA validation and benchmarking.
"""

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch


def run_c_binary(
    bin_path: str,
    B: int,
    H: int,
    S: int,
    D: int,
    seed: int,
    threads: int,
    warmup: int = 0,
    iters: int = 1,
    validate_outdir: Path | None = None,
    use_srun: bool = False,
) -> str:
    """
    Run the C binary with the given parameters.
    Returns stdout as a string.

    If use_srun=True, the binary is launched via 'srun' to ensure proper
    CPU affinity binding in SLURM environments.
    """
    # Prepend srun if requested (for SLURM environments)
    cmd = ["srun"] if use_srun else []

    cmd.extend(
        [
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
    )
    if validate_outdir is not None:
        cmd.extend(["--validate-outdir", str(validate_outdir)])

    return subprocess.check_output(cmd, text=True)


def _load_tensor(path: Path, shape: tuple) -> torch.Tensor:
    """Load a binary float32 tensor from disk as a contiguous torch.Tensor."""
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape)).contiguous()


def load_artifacts(
    outdir: Path,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load all artifacts from C binary output directory.
    Returns (meta, Q, K, V, out_c).
    """
    with open(outdir / "meta.json", "r") as f:
        meta = json.load(f)

    B = int(meta["batch"])
    H = int(meta["n_heads"])
    S = int(meta["seq_len"])
    D = int(meta["head_dim"])
    shape = (B, H, S, D)

    Q = _load_tensor(outdir / "q.bin", shape)
    K = _load_tensor(outdir / "k.bin", shape)
    V = _load_tensor(outdir / "v.bin", shape)
    out_c = _load_tensor(outdir / "out.bin", shape)

    return meta, Q, K, V, out_c


def parse_c_time(output: str) -> float:
    """Extract per-iteration time in seconds from C binary output."""
    m = re.search(r"CPU attention forward \(per-iter\):\s*([0-9.]+)\s*s", output)
    if not m:
        raise RuntimeError(
            "Could not parse per-iter time from binary output.\nOutput was:\n" + output
        )
    return float(m.group(1))
