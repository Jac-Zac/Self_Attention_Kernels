= Introduction

Transformer models have had a major impact on many areas, especially natural language processing @vaswani2017attention. At the core of these models lies the attention mechanism, which enables dynamic, context-dependent weighting of input sequences. However, this mechanism is also the main computational bottleneck, particularly due to its quadratic scaling with respect to the context length. This makes self-attention, especially for long sequences, one of the most interesting problems to optimize.

This project will focus on different ways to optimize the *causal multi-head self-attention* (CMHA) operation, which is the variant used in autoregressive language models like GPT @radford2018improving @radford2019language @brown2020language. 
I develop and benchmark three different sets of progressively optimized implementations:

1. *Single-threaded CPU*: A baseline implementation and various improved versions leveraging different vectorization techniques.
2. *Multi-threaded CPU*: An OpenMP-parallelized version that exploits modern multi-core architectures.
3. *GPU (CUDA)*: Progressive versions of massively parallel implementation leveraging GPU memory hierarchies

For educational purposes: each implementation builds upon the previous one, addressing performance bottlenecks and leveraging as much of the hardware as possible. 

For benchmarking, I use a parameter setting meant to be representative of modern decoder-only LLM attention (context length and per-head dimension). In particular, I took inspiration from the OLMo 2 architecture as a reasonable reference point for these choices @teamolmo2024olmo2furious.

I validate correctness against PyTorch's reference implementation and analyze performance characteristics on different systems due to resource constraints. 
In particular each chapter provides the hardware details used for its measurements.

== Reproducibility

Repository: #link("https://github.com/Jac-Zac/Self_Attention_Kernels")[github.com/Jac-Zac/Self_Attention_Kernels].
All kernels, tests (vs PyTorch SDPA), and benchmarking scripts used for this report are included. 

Additionally an old additional branch named `no_allign` was used to perform some of the benchmakring in the first chapter.

== Hardware Platform

All CPU benchmarks in this work are conducted on the #link("https://orfeo-doc.areasciencepark.it/HPC/computational-resources/#logical-partitions")[GENOA partition] of the Orfeo HPC cluster. This partition comprises 13 AMD nodes, each equipped with two #link("https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9374f.html")[AMD EPYC 9374F] processors and 512 GB of RAM. Based on the Zen 4 architecture, these processors offer 32 cores per socket (64 cores per node with SMT disabled), with a high base clock speed of 3.85 GHz, making them well-suited for intensive CPU workloads.

Key specifications relevant to this study:
- *Architecture:* x86_64 (Zen 4)
- *Cores:* 64 per node (32 per socket $times$ 2 sockets)
- *Vector Extensions:* AVX-512 support
- *L3 Cache:* 256 MB shared per socket

== Scope of the implementation

The focus of this project is the implementation of the *causal scaled dot-product attention kernel*. 
The implementation operates directly on pre-projected query, key, and value tensors $bold(Q)$, $bold(K)$, and $bold(V)$, which are assumed to be already partitioned per attention head and laid out contiguously in memory.
As such, the linear input projections $bold(X) bold(W)_q$, $bold(X) bold(W)_k$, and $bold(X) bold(W)_v$ are not included.
Furthermore, this implementation computes attention independently for each head. 
Head concatenation, the output projection $bold(W)_o$, and residual connections are also intentionally omitted, as they can be performed separately from the attention kernel I will focus on.

== Motivation and Scope

The quadratic complexity of self-attention ($cal(O)(T^2 d)$ for sequence length $T$ and model dimension $d$) presents fundamental challenges:

- *Memory bandwidth limitations*: Naive implementations repeatedly transfer large matrices between main memory and compute units
- *Not SIMD optimize*: The computation doesn't leverage SIMD instructions effectivly in the naive implementations 

Recent work like FlashAttention @dao2022flashattention demonstrates that careful algorithm-hardware co-design can achieve significant speedups (2-4× for training, 10-20× for long-context inference). 
Though I will start by simply implementing versions from my previous knowledge and then building up to more and more sophisticated tricks.

== Additional Notes

// TODO:

I have to evaluate the feasibility of implmeenting all of this and possibly adding Ring Attentention @liu2023ringattentionblockwisetransformers where comunication would happen via MPI.

Though I will first implement SIMD and CUDA versions and then review the paper on Ring Attention.
