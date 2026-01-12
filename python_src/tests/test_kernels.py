"""Validate CMHSA kernel output against PyTorch reference using GPT-2 weights.

Uses cached Q/K/V tensors from conftest.py to avoid regenerating them
for each kernel version during `make test`.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
# Import configs from conftest to ensure consistency
from conftest import GPT2_CONFIGS
from utils import load_output_artifact, run_c_binary_with_input, save_qkv_artifacts

BIN = "./cmhsa.out"


@pytest.mark.parametrize("layer,text,desc", GPT2_CONFIGS)
def test_kernel_gpt2(layer, text, desc, qkv_cache, backend, tmp_path):
    """Validate kernel output against PyTorch using cached GPT-2 Q,K,V."""
    device = "cuda" if backend == "cuda" else "cpu"

    # Get cached tensors
    Q, K, V, out_ref = qkv_cache[desc]

    # Save to temp directory for C++ to read
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    save_qkv_artifacts(input_dir, Q, K, V)

    # Run C++ kernel
    run_c_binary_with_input(
        BIN,
        input_dir=input_dir,
        validate_outdir=output_dir,
    )

    # Load C++ output
    _, out_c = load_output_artifact(output_dir, device=device)

    # Move reference to correct device for comparison
    out_ref = out_ref.to(device)

    # Compare
    max_err = (out_c - out_ref).abs().max().item()
    mean_err = (out_c - out_ref).abs().mean().item()

    assert torch.allclose(
        out_c, out_ref, rtol=1e-4, atol=1e-5
    ), f"GPT-2 layer {layer} ({desc}): max_err={max_err:.2e}, mean_err={mean_err:.2e}"
