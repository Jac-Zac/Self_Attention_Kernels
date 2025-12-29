#!/usr/bin/env python3
"""
Shared utilities for CMHSA validation and benchmarking.
"""

import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_TMP = RESULTS_DIR / "tmp"


@contextmanager
def tmp_artifacts_dir():
    """Context manager that creates results/tmp and cleans it up on exit."""
    RESULTS_TMP.mkdir(parents=True, exist_ok=True)
    try:
        yield RESULTS_TMP
    finally:
        if RESULTS_TMP.exists():
            shutil.rmtree(RESULTS_TMP)


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

    Args:
        bin_path: Path to the compiled binary
        B: Batch size
        H: Number of attention heads
        S: Sequence length
        D: Head dimension
        seed: Random seed for reproducibility
        threads: Number of threads to use
        warmup: Number of warmup iterations (not timed)
        iters: Number of timed iterations
        validate_outdir: Optional directory to save validation artifacts
        use_srun: Whether to use srun for SLURM CPU affinity binding

    Returns:
        stdout: Complete standard output from the binary as a string
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
    """
    Load a binary float32 tensor from disk as a contiguous torch.Tensor.

    Args:
        path: Path to the binary file
        shape: Target shape for the tensor (B, H, S, D)

    Returns:
        torch.Tensor: Loaded and reshaped tensor
    """
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape)).contiguous()


def load_artifacts(
    outdir: Path,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load all artifacts from C binary output directory.

    Args:
        outdir: Directory containing meta.json and binary tensor files

    Returns:
        tuple: (meta, Q, K, V, out_c) where meta is a dict with config info
               and Q, K, V, out_c are torch.Tensors of shape (B, H, S, D)
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
    """
    Extract per-iteration time in seconds from C binary output.

    Args:
        output: Standard output from the C binary

    Returns:
        float: Per-iteration execution time in seconds

    Raises:
        RuntimeError: If the time pattern is not found in output
    """
    m = re.search(r"CPU attention forward \(per-iter\):\s*([0-9.]+)\s*s", output)
    if not m:
        raise RuntimeError(
            "Could not parse per-iter time from binary output.\nOutput was:\n" + output
        )
    return float(m.group(1))
