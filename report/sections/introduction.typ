= Introduction

Transformer models have had a major impact on many areas, especially natural language processing @vaswani2017attention. At the core of these models lies the attention mechanism, which enables dynamic, context-dependent weighting of input sequences. However, this mechanism is also the main computational bottleneck, particularly due to its quadratic scaling with respect to the context length. This makes self-attention, especially for long sequences, one of the most interesting problems to optimize.

This project will focus on different ways to optimize the *causal multi-head self-attention* (CMHA) operation, which is the variant used in autoregressive language models like GPT @radford2018improving @radford2019language @brown2020language. 
I develop and benchmark three different sets of progressively optimized implementations:

1. *Single-threaded CPU*: A baseline implementation and various improved versions leveraging different vectorization techniques.
2. *Multi-threaded CPU*: An OpenMP-parallelized version that exploits modern multi-core architectures.
3. *GPU (CUDA)*: Progressive versions of massively parallel implementation leveraging GPU memory hierarchies

For educational purposes: each implementation builds upon the previous one, addressing performance bottlenecks and leveraging as much of the hardware as possible. 
I validate correctness against PyTorch's reference implementation and analyze performance characteristics on ... (I have to decide where to run it)

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
