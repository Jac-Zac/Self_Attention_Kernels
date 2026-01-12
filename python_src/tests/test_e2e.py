"""End-to-end tests validating CMHSA kernel across full GPT-2 model.

Tests that substituting PyTorch's attention with our kernel produces identical
model outputs across all layers.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import GPT2_CONFIGS, _detect_backend
from utils import load_output_artifact, run_c_binary_with_input, save_qkv_artifacts

BIN = "./cmhsa.out"


def _run_layer_with_kernel(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    input_dir: Path,
    output_dir: Path,
    device: str,
) -> torch.Tensor:
    """Run a single layer through our kernel and return output."""
    save_qkv_artifacts(input_dir, Q, K, V)
    run_c_binary_with_input(BIN, input_dir=input_dir, validate_outdir=output_dir)
    _, out_c = load_output_artifact(output_dir, device=device)
    return out_c


def _run_full_model_with_kernel(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Run full GPT-2 model using our kernel for all attention layers."""
    device = "cuda" if backend == "cuda" else "cpu"
    input_ids = input_ids.to(device)

    with torch.no_grad():
        hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=device)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for layer_idx, block in enumerate(model.transformer.h):
                B, S, D = hidden_states.shape
                attn = block.attn
                n_heads = attn.num_heads
                head_dim = attn.head_dim

                # Layer norm before attention (GPT-2 uses pre-LayerNorm)
                normed = block.ln_1(hidden_states)

                # Compute Q, K, V
                qkv = attn.c_attn(normed)
                q, k, v = qkv.split(attn.split_size, dim=2)

                q = q.view(B, S, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()
                k = k.view(B, S, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()
                v = v.view(B, S, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()

                # Run our kernel
                input_dir = tmpdir_path / f"layer{layer_idx}_input"
                output_dir = tmpdir_path / f"layer{layer_idx}_output"
                attn_out = _run_layer_with_kernel(
                    q, k, v, input_dir, output_dir, device
                )

                # Reshape back to (B, S, D)
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, S, -1)

                # Projection and residual
                attn_out = attn.c_proj(attn_out)
                hidden_states = hidden_states + attn_out

                # Layer norm before MLP (GPT-2 uses pre-LayerNorm)
                normed = block.ln_2(hidden_states)

                # MLP (GPT-2 uses tanh approximation of GELU)
                mlp_out = block.mlp.c_fc(normed)
                mlp_out = F.gelu(mlp_out, approximate="tanh")
                mlp_out = block.mlp.c_proj(mlp_out)
                hidden_states = hidden_states + mlp_out

        # Layer norm
        hidden_states = model.transformer.ln_f(hidden_states)

        # LM head
        logits = model.lm_head(hidden_states)

    return logits


@pytest.mark.parametrize(
    "text,desc",
    [
        (
            "In recent years, deep learning has transformed the field of natural language processing. The introduction of the transformer architecture marked a significant milestone, enabling models to handle long-range dependencies more effectively than previous approaches. Self-attention mechanisms allow models to weigh the importance of different words in a sentence when processing each word, leading to better contextual representations. This has led to remarkable improvements in machine translation, text summarization, question answering, and many other NLP tasks. Large language models trained on massive datasets have demonstrated impressive capabilities in generating coherent text, answering questions, and even performing multi-step reasoning tasks.",
            "long_sequence",
        )
    ],
)
def test_e2e_model_output(text, desc, backend):
    """Validate that our kernel produces identical model outputs."""
    backend_detected = _detect_backend()
    if backend_detected == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA binary but no GPU available")

    device = "cuda" if backend_detected == "cuda" else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_pytorch = GPT2LMHeadModel.from_pretrained("gpt2")
    model_pytorch.to(device)
    model_pytorch.eval()

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # PyTorch reference
    with torch.no_grad():
        logits_pytorch = model_pytorch(input_ids).logits

    # Our kernel
    logits_kernel = _run_full_model_with_kernel(
        model_pytorch, input_ids, backend_detected
    )

    # Compare (tolerances are slightly higher than single-layer tests due to
    # error accumulation across 12 transformer layers)
    max_err = (logits_kernel - logits_pytorch).abs().max().item()
    mean_err = (logits_kernel - logits_pytorch).abs().mean().item()

    assert torch.allclose(logits_kernel, logits_pytorch, rtol=1e-3, atol=1e-4), (
        f"E2E test ({desc}): max_err={max_err:.2e}, mean_err={mean_err:.2e}"
    )
