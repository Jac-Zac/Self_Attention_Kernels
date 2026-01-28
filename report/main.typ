#import "@preview/red-agora:0.1.2": project

#show: project.with(
  title: "Tutorial: Implementing Efficient Causal Multi-Head Self-Attention for CPU and GPU",
  subtitle: "A Performance-Oriented Study of Self-Attention Kernels Across Computing Architectures",
  authors: (
    "Jacopo Zacchigna",
  ),
  school-logo: [],
  company-logo: [],
  mentors: (
    "Prof. Luca Tornatore
",
  ),
  footer-text: "DSAI",
  branch: "Advanced High Performance Computing",
  academic-year: "2025-2026",)

// Enable equation numbering and justify
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: set text(fill: blue)

#include "sections/introduction.typ"
#include "sections/background.typ"
#include "sections/single_thread.typ"
#include "sections/multi_thread.typ"
#include "sections/cuda.typ"

= Conclusion and Future Work

This work presented a systematic exploration of causal multi-head self-attention kernel optimization across three computing platforms: single-threaded CPU, multi-threaded CPU with OpenMP, and GPU with CUDA.
Each implementation progressed through multiple versions, revealing the distinct optimization strategies required for different hardware architectures.

== Summary of Results

=== Single-Threaded CPU

The single-threaded implementation demonstrated the critical importance of enabling compiler auto-vectorization.
Starting from a naive baseline (v0) that was 10× slower than PyTorch's naive implementation, we achieved a 30× speedup through:

- *Loop restructuring* (v1): Respecting causal masking during computation and reordering loops to make the head dimension innermost, enabling SIMD vectorization
- *Compiler flags* (v1_b-d): Enabling `-ffast-math` subset flags to permit associative floating-point operations, unlocking true vector accumulation with `vfmadd231ps` instructions
- *Memory alignment* (v1_c): Padding strides to 64-byte boundaries for aligned AVX-512 loads

The final single-threaded kernel (v2) achieves approximately 2.8× speedup over PyTorch naive and reaches 49% of PyTorch SDPA performance—a reasonable result for a single-threaded implementation compared to PyTorch's multi-threaded backend.

=== Multi-Threaded CPU

The OpenMP implementation scaled efficiently across cores, with v1 achieving near-ideal speedup up to 64 threads on a 128-core system.
Key optimizations included:

- *Query tiling* (v1): Processing tiles of 32 queries together, allowing K and V rows to be loaded once per tile rather than once per query
- *Cache-aware blocking*: Ensuring tiles fit in L3 cache to maximize data reuse

At 128 threads, the v1 kernel matches PyTorch SDPA performance and achieves 94× speedup over the single-threaded baseline.
The multi-threaded CPU implementation represents the most mature optimization in this work, approaching production-level performance through careful attention to cache hierarchies and memory access patterns.

 === CUDA GPU

 The GPU implementation revealed both the potential and the challenges of massively parallel attention computation.
 Progressive optimizations achieved:

 - *Warp-level parallelism* (v1): ~8–9× speedup through collaborative dot products and coalesced memory access
 - *Multi-warp blocks* (v2): Additional ~1.4× from better occupancy and XOR-based reductions
 - *Online softmax* (v3): Eliminated workspace memory entirely and reduced global-memory round trips

The fastest kernel `v6` in runs (0.11642 s). Relative to PyTorch:

 - `v6` is ≈1.11× slower than PyTorch naive (0.10525 s) in our additional runs
 - `v6` is ≈5.6× slower than PyTorch SDPA (0.02078 s)

The gap with PyTorch SDPA is significant, closing the gap further probably requires Tensor Core integration, head-dimension specialization, and more aggressive shared-memory tiling.

== Future Work

Several directions could extend this work:

=== GPU Optimizations

The most impactful improvements for the CUDA implementation would be:

1. *Shared memory tiling for K and V*: Loading blocks of K and V into shared memory for reuse across multiple warps, following the Flash Attention approach. This is a work in progress inside v5 - v6. 

2. *Tensor Core integration*: Leveraging `wmma` or `mma` instructions for matrix multiply-accumulate operations. This requires restructuring to operate on FP16 inputs and matrix tiles, but would provide 8-16× throughput improvement for the Q·K and attention·V computations.

=== Multi-GPU Scaling

For sequences exceeding single-GPU memory or requiring distributed inference, Ring Attention @liu2023ringattentionblockwisetransformers offers an elegant solution.
The key insight is that attention can be computed block-wise, with K/V blocks circulated around a ring of GPUs while Q blocks remain stationary.
This overlaps communication with computation and achieves near-linear scaling.

An MPI-based implementation using CUDA-aware MPI for direct GPU-to-GPU transfers would be a natural extension of this work, combining the per-GPU optimizations developed here with distributed memory parallelism.

== Concluding Remarks

This project demonstrates that achieving high performance in self-attention requires deep understanding of both the algorithm and the target hardware.
The same mathematical operation—scaled dot-product attention—demands fundamentally different implementation strategies on CPUs versus GPUs, and even within GPUs, the gap between a correct implementation and a fast one spans orders of magnitude.

The educational value of this exercise lies not in matching production performance, but in understanding *why* certain optimizations matter and *how* hardware constraints shape algorithmic choices.
The progression from naive baselines to reasonably optimized kernels illustrates the key principles: vectorization and cache locality on CPUs, coalescing and occupancy on GPUs, and numerical stability everywhere.

All code, benchmarks, and this report are available at #link("https://github.com/Jac-Zac/Self_Attention_Kernels")[github.com/Jac-Zac/Self_Attention_Kernels] for reference and further experimentation.

= References
#bibliography("refs.bib")

#include "sections/appendix.typ"
