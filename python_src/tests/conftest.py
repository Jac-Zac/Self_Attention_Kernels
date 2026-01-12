"""Pytest configuration for kernel validation tests.

Caches GPT-2 Q/K/V tensors and reference outputs in .pytest_cache to avoid
regenerating them for each kernel version during `make test`.
"""

import hashlib
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Test configurations: (layer, text, description)
# Duplicated here to enable caching before test collection
GPT2_CONFIGS = [
    (0, "Hello", "short_layer0"),
    (0, "The quick brown fox jumps over the lazy dog.", "medium_layer0"),
    (6, "Attention is all you need.", "layer6"),
    (11, "The transformer architecture revolutionized NLP.", "last_layer"),
    (
        0,
        "In recent years, deep learning has transformed the field of natural language processing. The introduction of the transformer architecture marked a significant milestone, enabling models to handle long-range dependencies more effectively than previous approaches. Self-attention mechanisms allow models to weigh the importance of different words in a sentence when processing each word, leading to better contextual representations. This has led to remarkable improvements in machine translation, text summarization, question answering, and many other NLP tasks. Large language models trained on massive datasets have demonstrated impressive capabilities in generating coherent text, answering questions, and even performing multi-step reasoning tasks.",
        "long_sequence",
    ),
]


def _get_cache_key() -> str:
    """Generate a stable cache key based on test configurations."""
    config_str = str(GPT2_CONFIGS)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def _get_cache_dir(rootdir: Path) -> Path:
    """Get the cache directory for QKV tensors."""
    return rootdir / ".pytest_cache" / "qkv_cache" / _get_cache_key()


def _extract_qkv(
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


def _detect_backend(bin_path: str = "./cmhsa.out") -> str:
    """Run the binary with minimal args to detect backend from output."""
    try:
        output = subprocess.check_output(
            [
                bin_path,
                "--batch",
                "1",
                "--n_heads",
                "1",
                "--seq_len",
                "1",
                "--head_dim",
                "16",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        if "CUDA" in output:
            return "cuda"
        return "cpu"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "cpu"


@pytest.fixture(scope="session")
def backend():
    """Auto-detect and return the backend from the compiled binary."""
    return _detect_backend()


@pytest.fixture(scope="session")
def qkv_cache(
    request,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load or generate cached Q/K/V tensors and PyTorch reference outputs.

    Caches tensors in .pytest_cache/qkv_cache/<hash>/ to persist across
    pytest invocations (e.g., during `make test` which runs pytest multiple times).

    Returns:
        dict mapping config description to (Q, K, V, out_ref) tuple
    """
    cache_dir = _get_cache_dir(Path(request.config.rootdir))

    # Check if cache is complete
    cache_complete = cache_dir.exists() and all(
        (cache_dir / f"{desc}.pt").exists() for _, _, desc in GPT2_CONFIGS
    )

    if cache_complete:
        # Load from cache
        results = {}
        for _, _, desc in GPT2_CONFIGS:
            data = torch.load(cache_dir / f"{desc}.pt", weights_only=True)
            results[desc] = (data["Q"], data["K"], data["V"], data["out_ref"])
        return results

    # Generate and cache
    print("\n[pytest] Generating QKV cache (first run or cache cleared)...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    results = {}
    for layer, text, desc in GPT2_CONFIGS:
        Q, K, V = _extract_qkv(model, tokenizer, text, layer)
        out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Save to cache
        torch.save(
            {"Q": Q, "K": K, "V": V, "out_ref": out_ref}, cache_dir / f"{desc}.pt"
        )
        results[desc] = (Q, K, V, out_ref)

    print(f"[pytest] QKV cache saved to {cache_dir}")
    return results


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests if CUDA binary detected but no GPU available."""
    backend = _detect_backend()
    if backend == "cuda" and not torch.cuda.is_available():
        skip_marker = pytest.mark.skip(reason="CUDA binary but no GPU available")
        for item in items:
            item.add_marker(skip_marker)
