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


def _load_tensor(path: Path, shape: tuple) -> torch.Tensor:
    """Load a binary float32 file into a torch.Tensor with given shape."""
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape)).contiguous()


def load_output_artifact(
    outdir: Path,
    device: str = "cpu",
) -> tuple[dict, torch.Tensor]:
    """Load meta.json and output tensor from C binary output directory."""
    with open(outdir / "meta.json", "r") as f:
        meta = json.load(f)

    B = int(meta["batch"])
    H = int(meta["n_heads"])
    S = int(meta["seq_len"])
    D = int(meta["head_dim"])
    shape = (B, H, S, D)

    device_obj = torch.device(device)
    out_c = _load_tensor(outdir / "out.bin", shape).to(device_obj)

    return meta, out_c


def parse_gpu_info(output: str) -> dict:
    """Extract GPU info (name, compute_capability, memory_gb, sm_count) from output."""
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
    """Extract per-iteration time in seconds from C binary output."""
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


def run_c_binary_with_input(
    bin_path: str,
    input_dir: Path,
    warmup: int = 0,
    iters: int = 1,
    validate_outdir: Path | None = None,
    use_srun: bool = False,
) -> str:
    """Run the C/CUDA binary with Q,K,V loaded from input_dir. Returns stdout."""
    cmd = ["srun"] if use_srun else []

    base_cmd = [
        bin_path,
        "--input-dir",
        str(input_dir),
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


def save_qkv_artifacts(
    outdir: Path,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> None:
    """Save Q, K, V tensors and metadata to directory for C++ to read."""
    outdir.mkdir(parents=True, exist_ok=True)

    B, H, S, D = Q.shape
    meta = {
        "batch": B,
        "n_heads": H,
        "seq_len": S,
        "head_dim": D,
        "dtype": "float32",
    }

    Q.contiguous().numpy().tofile(outdir / "q.bin")
    K.contiguous().numpy().tofile(outdir / "k.bin")
    V.contiguous().numpy().tofile(outdir / "v.bin")

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
