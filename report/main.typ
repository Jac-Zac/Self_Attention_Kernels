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
  academic-year: "2024-2025",
)

// Enable equation numbering and justify
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: set text(fill: blue)

#include "sections/introduction.typ"
#include "sections/background.typ"

= Single-Threaded CPU Implementation

Single-threaded kernels provide the cleanest environment to study performance bottlenecks such as SIMD utilization, cache behavior, and the cost of the softmax. 
I begin with a correctness-first baseline (v0), then introduce incremental changes (v1, v2, ...) that improve vectorization and reduce memory traffic.
To evaluate the correctes of different implementations they are always compared to the actual corresponding PyTorch implementation.
The compilation flags will be extremly important in the following sections everything though will be compiled at least with `-O3 -march=native`

== Benchmark Configuration

For the single threaded version the benchmarks are run with the following configuration, inspired by the OLMo 2 architecture @teamolmo2024olmo2furious but slightly scaled down:

`batch=1`, `n_heads=4`, `seq_len=4096`, `head_dim=128`, `seed=1337`, `warmup=5`, `iters=25`.

This takes a reasonable amount of time on the GENOA partition and results are averaged across iterations. 

Code: #link("https://github.com/Jac-Zac/Self_Attention_Kernels/tree/main/kernels/single_thread")[github.com/Jac-Zac/Self_Attention_Kernels/tree/main/kernels/single_thread].

== Simple baseline implementation (v0)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/single_thread/v0.cpp")[v0] is a straightforward, correctness-first baseline implementation of causal scaled dot-product attention. 
It closely follows the textbook computation ($Q K^T$, causal masking, numerically-stable softmax, then the weighted sum with $V$) and is intentionally written for clarity rather than performance. 
The structure is heavily inspired by #link("https://github.com/HicrestLaboratory/Open-VIT-bench/blob/976c148a3fdda227918df4a8419e8cb25ef7ff89/omp_src/attention.cpp#L170")[this reference implementation] though it uses a numerically-stable softmax.
Notably this version already uses the padded constructions of $Q, K, V$ which will come in especially handy for later versions. The implementation follows the following steps:

1. Compute all $Q K^T$ scores (including masked positions)
2. Apply causal mask by setting future positions to $-infinity$
3. Compute softmax using numerically stable two-pass algorithm
4. Explicitly zero out masked positions after softmax
5. Compute weighted sum of values

== Improved kernel structure (v1)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/single_thread/v1.cpp")[v1] is an improvement on the previous version which has the 2 following properities:

1. Respecting the causal mask during computation (not computing masked values)
2. Changing the loop order in the output computation to improve cache locality (making `head_dim` the innermost loop to also allows for better vectorization)

Benchmarks for *v1* are split into the following subversions to isolate the impact of different optimizations:

1. *v1_a:* uses no additional compilation flags beyond `-O3 -march=native`
2. *v1_b:* enables selective `-ffast-math` flags: `-fassociative-math -fno-trapping-math -ffinite-math-only -fno-signed-zeros` to allow for autovectorization to actually take full effect
3. *v1_c:* pads `head_dim` to ensure proper alignment, allowing the compiler to emit aligned vectorized instructions (this padding is retained in all subsequent versions)
4. *v1_d:* enables full `-ffast-math`, which allows vectorization of transcendental functions like `expf`

This implementation already provides roughly a 5x speedup compared to v0. Moving from v1_a to v1_b yields an additional 3.5x gain, while v1_c provides a further 30% improvement through aligned memory access. Enabling full `-ffast-math` in v1_d adds another 20% on top of these gains.

// 3. *v1_d:* enables the additiaonal flag `-mprefer-vector-width=512` to hint the compiler on actually using avx512

=== Deeper analyzes

To understand why those optimization were performed we can start digging in the assembly. First off all we notice that the compiler is telling ous it is actually able to vectorize the dot product loop. But looking at it more carefully we notice something strange.

Although the compiler appears to vectorize the naive solution based on its outputs, the generated code is not fully vectorized. 
One clear example can be seen in the self-attention operation.

```c
// V1.cpp approx line 50
dot_product += Q[query_offset + d] * K[key_offset + d];
```

Generates the following assembly:

I need to be more precise where ! The dot product I believe actually vectorizes correctly (To review)
- Other parts defenitly do not vecotize like all of the reductions

```asm
; 1. GOOD: It uses AVX (YMM, 256-bit) to LOAD and MULTIPLY 8 floats at once.
vmovups ymm2, YMMWORD PTR [rdi+rax]       ; Load 8 floats from Q
vmulps  ymm2, ymm2, YMMWORD PTR [rdx+rax] ; Multiply by 8 floats from K (Packed Single)

; 2. BAD: It immediately reduces those 8 floats to a single scalar inside the loop.
vaddss  xmm0, xmm0, xmm2                  ; Add lowest float to accumulator (Scalar Single)
vshufps xmm1, xmm2, xmm2, 85              ; Shuffle to get next float...
vaddss  xmm0, xmm0, xmm1                  ; Add...
vunpckhps xmm1, xmm2, xmm2                ; Unpack high parts...
; ... continuous shuffling and scalar adding ...
```

be careful of things like this:

```asm
vaddss xmm0, xmm0, xmm1
vaddss xmm0, xmm0, xmm3
vaddss xmm0, xmm0, xmm3
vaddss xmm0, xmm0, xmm1
```

I don't want to then do scalar reduction

Instead of keeping YMM registers as partial sums and reducing them afterward, the compiler immediately reduces to scalar operations inside the loop, providing no speedup. 
This occurs because we use floating-point values, and GCC prioritizes *IEEE 754 compliance* without additional flags. 
Since addition is not associative $(a + b) + c != a + (b + c)$ the compiler must revert to scalar instructions.

Another case where this occurs is ... is the softmax show the differennnce between v0 and v1 !

== Remaining versions

Make a comment on what the reaming versions try to do.

Like fusing things dons't have much of an effect and also. and hinting with opmp dones't have any difference since we are already doing fully vectorized computation with fastmath though it is also needed for the actual vectorization of the expf function thus it is going to be kept.

= Multi-Threaded CPU Implementation  

[To be completed in next sections]

= CUDA Implementation

[To be completed in next sections]

= Performance Analysis and Results

[To be completed in next sections]

= Conclusion and Future Work

[To be completed in next sections]

= References
#bibliography("refs.bib")
