import time

import cmhsa

import torch


# ------------------------------------------------------------
# Reference PyTorch causal attention
# ------------------------------------------------------------
def reference_attention(Q, K, V):
    """
    Q, K, V: [B, H, T, D]
    """
    B, H, T, D = Q.shape
    scale = D**-0.5

    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    return torch.matmul(P, V)


# ------------------------------------------------------------
# Correctness test
# ------------------------------------------------------------
def check_correctness(
    batch=2,
    n_heads=4,
    seq_len=128,
    head_dim=64,
    dtype=torch.float32,
):
    print("Running correctness check...")
    print(f"  dtype={dtype}, head_dim={head_dim}")

    torch.manual_seed(0)

    Q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    with torch.no_grad():
        out_ref = reference_attention(Q, K, V)
        out_cuda = cmhsa.forward(Q, K, V)

    max_err = (out_ref - out_cuda).abs().max().item()
    mean_err = (out_ref - out_cuda).abs().mean().item()

    print(f"  max error : {max_err:.6e}")
    print(f"  mean error: {mean_err:.6e}")

    if dtype == torch.float32:
        assert max_err < 1e-4, "FP32 error too large"
    else:
        assert max_err < 3e-3, "FP16 error too large"

    print("Correctness OK.\n")


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
def benchmark(
    batch=4,
    n_heads=8,
    seq_len=256,
    head_dim=64,
    dtype=torch.float16,
    iters=100,
):
    print("Running benchmark...")
    print(f"  B={batch}, H={n_heads}, T={seq_len}, D={head_dim}, dtype={dtype}")

    Q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # Warmup
    for _ in range(10):
        cmhsa.forward(Q, K, V)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        cmhsa.forward(Q, K, V)
    torch.cuda.synchronize()
    t1 = time.time()

    ms = (t1 - t0) * 1000 / iters
    print(f"  CMHSA forward: {ms:.3f} ms")

    # PyTorch baseline
    for _ in range(10):
        reference_attention(Q, K, V)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        reference_attention(Q, K, V)
    torch.cuda.synchronize()
    t1 = time.time()

    ms_ref = (t1 - t0) * 1000 / iters
    print(f"  PyTorch ref  : {ms_ref:.3f} ms")
    print(f"  Speedup     : {ms_ref / ms:.2f}x\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False

    # Correctness
    check_correctness(head_dim=64, dtype=torch.float32)
    check_correctness(head_dim=128, dtype=torch.float32)

    check_correctness(head_dim=64, dtype=torch.float16)
    check_correctness(head_dim=128, dtype=torch.float16)

    # Benchmark
    benchmark(seq_len=128, head_dim=64, dtype=torch.float16)
    benchmark(seq_len=256, head_dim=64, dtype=torch.float16)
