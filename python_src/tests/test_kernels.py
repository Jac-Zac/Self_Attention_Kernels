"""Validate CMHSA kernel output against PyTorch reference using GPT-2 weights."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_output_artifact, run_c_binary_with_input, save_qkv_artifacts

BIN = "./cmhsa.out"

# Test configurations: (layer, text, description)
GPT2_CONFIGS = [
    (0, "Hello", "short_layer0"),
    (0, "The quick brown fox jumps over the lazy dog.", "medium_layer0"),
    (6, "Attention is all you need.", "layer6"),
    (11, "The transformer architecture revolutionized NLP.", "last_layer"),
]


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 model once for all tests."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def extract_qkv(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Q, K, V from a specific GPT-2 layer."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    captured = {}

    def hook(module, args, output):
        hidden_states = args[0]
        attn = module

        qkv = attn.c_attn(hidden_states)
        q, k, v = qkv.split(attn.split_size, dim=2)

        batch_size = q.shape[0]
        q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, attn.num_heads, attn.head_dim)
        v = v.permute(0, 2, 1, 3)

        captured["Q"] = q.detach().clone()
        captured["K"] = k.detach().clone()
        captured["V"] = v.detach().clone()

    target_attn = model.transformer.h[layer].attn
    handle = target_attn.register_forward_hook(hook)

    with torch.no_grad():
        model(input_ids)

    handle.remove()

    return captured["Q"].float(), captured["K"].float(), captured["V"].float()


@pytest.mark.parametrize("layer,text,desc", GPT2_CONFIGS)
def test_kernel_gpt2(layer, text, desc, gpt2_model, backend, tmp_path):
    """Validate kernel output against PyTorch using real GPT-2 Q,K,V."""
    device = "cuda" if backend == "cuda" else "cpu"
    tokenizer, model = gpt2_model

    # Extract Q, K, V from GPT-2
    Q, K, V = extract_qkv(model, tokenizer, text, layer)

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

    # Compute PyTorch reference
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Compare
    max_err = (out_c - out_ref).abs().max().item()
    mean_err = (out_c - out_ref).abs().mean().item()

    assert torch.allclose(
        out_c, out_ref, rtol=1e-4, atol=1e-5
    ), f"GPT-2 layer {layer} ({desc}): max_err={max_err:.2e}, mean_err={mean_err:.2e}"
