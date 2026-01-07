= Multi-Threaded CPU Implementation  

The multi-threaded benchmarks were run on Orfeo's EPYC partition.

Benchmark configuration used in this chapter:
`BENCH_BATCH=4`, `BENCH_HEADS=32`, `BENCH_SEQLEN=4096`, `BENCH_ITERS=10`.

Code: #link("https://github.com/Jac-Zac/Self_Attention_Kernels/tree/main/kernels/multi_thread")[github.com/Jac-Zac/Self_Attention_Kernels/tree/main/kernels/multi_thread].

== Basic Parallelization (v0)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/multi_thread/v0.cpp")[v0] implements straightforward multi-threading by parallelizing over the batch and heads dimensions.
The parallelization strategy uses `collapse(2)` to create 4 × 32 = 128 independent work units (one for each batch-head pair), providing good load balance across threads.
Each thread allocates its own scratch space for attention weights using `thread_id * seq_len_padded`, avoiding contention between threads.
The implementation follows the same algorithm as the optimized single-threaded v1, including fused max-finding with score computation, causal masking, and multiplication by inverse for normalization.
The benchmarks use OpenMP environment variables `OMP_PLACES=cores` and `OMP_PROC_BIND=close`.

Despite inheriting all single-threaded optimizations (aligned memory access, padded strides, and full `-ffast-math` vectorization), v0 still somewhat lags behind the most efficient implementation.
With the larger problem size (4096 sequence length), memory traffic becomes a big bottleneck, motivating the tiling approach in v1.

== Q-Tiling for Cache Reuse (v1)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/multi_thread/v1.cpp")[v1] introduces query tiling to improve cache locality and reduce memory traffic.
The key idea is to process `TILE_Q = 32` queries together, allowing each key row to be loaded once and reused for all queries in the tile that need it.

The algorithm consists of three passes:

1. *Pass 1:* Compute $Q K^T$ scores for the tile.
   By iterating over keys in the outer loop, each K row is loaded once per tile instead of once per query, reducing K memory traffic.
   Causal masking is handled by `q_start = max(key_pos, q_tile)`, which skips masked positions.

2. *Pass 2:* Numerically stable softmax.
   Each query computes its own max and sum of exponentials independently.

3. *Pass 3:* Weighted V accumulation.
   Same loop pattern as Pass 1, with each V row loaded once per tile.

The implementation adds `ASSUME_ALIGNED` hints on rows to improve vectorization, similar to the single-threaded versions.

=== Strong Scaling Results

The strong scaling benchmark was run on Orfeo's EPYC partition (AMD EPYC 9654, 128 cores) with the following configuration: `batch=4`, `n_heads=32`, `seq_len=4096`, `iters=10`.
Thread counts were varied from 1 to 128 to evaluate parallel efficiency.

#figure(
  image("../figures/benchmark_strong_scaling.png", width: 95%),
  caption: [Strong scaling results for multi-threaded kernels. *Left:* speedup relative to single-threaded baseline (ideal scaling shown as dashed line). *Right:* execution time.],
) <fig:benchmark_strong_scaling>

Detailed results are available in @tab:benchmark_strong_scaling in the appendix. The v1 kernel achieves near-ideal scaling up to 64 threads, with a speedup of approximately 47x over the single-threaded baseline.
At 128 threads, v1 reaches timing comparable to PyTorch's highly optimized SDPA implementation and approximately 19x faster than naive PyTorch.

Compared to v0, v1 shows significantly better scaling due to reduced memory traffic from query tiling and better cache reuse.
At 128 threads, v1 outperforms v0 by approximately 2.3x, demonstrating the effectiveness of cache-aware optimizations for large problem sizes. 
