"""Validate CMHSA kernel output against PyTorch reference."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_artifacts, run_c_binary

BIN = "./cmhsa.out"

# (batch, n_heads, seq_len, head_dim)
CONFIGS = [
    (4, 8, 16, 32),  # default
    (1, 1, 4, 16),  # minimal
    (2, 4, 64, 64),  # larger
]


@pytest.mark.parametrize("batch,n_heads,seq_len,head_dim", CONFIGS)
def test_kernel(batch, n_heads, seq_len, head_dim, backend, tmp_path):
    """Validate kernel output against PyTorch scaled_dot_product_attention."""
    device = "cuda" if backend == "cuda" else "cpu"

    run_c_binary(
        BIN,
        batch,
        n_heads,
        seq_len,
        head_dim,
        seed=1337,
        threads=1,
        validate_outdir=tmp_path,
    )
    _, Q, K, V, out_c = load_artifacts(tmp_path, device=device)
    out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    max_err = (out_c - out_ref).abs().max().item()
    assert torch.allclose(
        out_c, out_ref, rtol=1e-4, atol=1e-5
    ), f"max_err={max_err:.6g}"
