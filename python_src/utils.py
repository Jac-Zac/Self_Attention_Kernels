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
    Run the C/CUDA binary with the given parameters.
    Automatically detects if binary is CUDA and omits --threads flag.
    """
    cmd = ["srun"] if use_srun else []

    base_cmd = [
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
    ]

    cmd.extend(base_cmd)

    if validate_outdir is not None:
        cmd.extend(["--validate-outdir", str(validate_outdir)])

    output = subprocess.check_output(cmd, text=True)

    return output


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


def parse_gpu_info(output: str) -> dict:
    """
    Extract GPU information from CUDA binary output.

    Args:
        output: Standard output from the C binary

    Returns:
        dict: GPU information with keys: name, compute_capability, memory_gb, sm_count
    """
    gpu_info = {}

    # Extract GPU name
    m = re.search(r"GPU Device:\s*(.+)", output)
    if m:
        gpu_info["name"] = m.group(1).strip()

    # Extract compute capability
    m = re.search(r"GPU Compute Capability:\s*(\d+)\.(\d+)", output)
    if m:
        gpu_info["compute_capability"] = f"{m.group(1)}.{m.group(2)}"

    # Extract memory
    m = re.search(r"GPU Memory:\s*([0-9.]+)\s*GB", output)
    if m:
        gpu_info["memory_gb"] = float(m.group(1))

    # Extract SM count
    m = re.search(r"GPU SM Count:\s*(\d+)", output)
    if m:
        gpu_info["sm_count"] = int(m.group(1))

    return gpu_info


def parse_c_time(output: str) -> float:
    """
    Extract per-iteration time in seconds from C binary output.

    Args:
        output: Standard output from the C binary

    Returns:
        float: Per-iteration execution time in seconds

    Raises:
        RuntimeError: If time pattern is not found in output
    """
    # Try CPU pattern first, then CUDA pattern
    m = re.search(
        r"(CPU|CUDA) attention forward \(per-iter\):\s*([0-9.]+)\s*(ms|s)", output
    )
    if not m:
        raise RuntimeError(
            "Could not parse per-iter time from binary output.\nOutput was:\n" + output
        )
    time_value = float(m.group(2))
    unit = m.group(3)
    # Convert ms to s if needed
    return time_value / 1000.0 if unit == "ms" else time_value
