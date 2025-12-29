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
The implementation follows the following steps:

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
3. *v1_c:* pads `head_dim` to ensure proper alignment, this (padding is retained in all subsequent versions)
4. *v1_d:* enables full `-ffast-math`, which allows vectorization of functions like `expf`

The `v1_a` implementation already provides roughly a 5x speedup compared to v0. Moving from v1_a to v1_b yields an additional 3.5x gain, while v1_c provides a further 30% improvement through aligned memory access. Enabling full `-ffast-math` in v1_d adds another 20% on top of these gains.

=== Deeper Analysis

To understand the performance differences between v1 subversions, we examine the generated assembly. The compiler reports successful vectorization, but the reality is more nuanced.

==== v1_a: Vectorized Loads with Scalar Reduction

With only `-O3 -march=native`, the compiler reports:

```
kernels/single_thread/v1.cpp:58:32: optimized: loop vectorized using 64 byte vectors
kernels/single_thread/v1.cpp:26:6: note: vectorized 4 loops in function.
```

The dot product loop (line 58-59):

```c
dot_product += Q[query_offset + d] * K[key_offset + d];
```

Generates AVX-512 code that *appears* because of the use of *zmm\** registers vectorized, but performs poorly for some reason.
Looking more carefully at the assembly we see the following:

```asm
; GOOD: Uses AVX-512 (ZMM) to LOAD and MULTIPLY 16 floats at once
vmovups zmm5, ZMMWORD PTR [rdi+rax]       ; Load 16 floats from Q
vmulps  zmm5, zmm5, ZMMWORD PTR [rdx+rax] ; Multiply by 16 floats from K

; BAD: Immediately reduces those 16 floats to scalar INSIDE the loop
vaddss  xmm0, xmm0, xmm5               ; Add lowest float to accumulator
vshufps xmm1, xmm5, xmm5, 85           ; Shuffle to get next float...
vaddss  xmm0, xmm0, xmm1               ; Add scalar...
vunpckhps xmm1, xmm5, xmm5             ; Unpack high parts...
vaddss  xmm0, xmm0, xmm1               ; Add scalar...
; ... 16 total vaddss instructions per iteration
```

Despite using 512-bit loads and multiplies, the compiler immediately reduces to scalar additions *inside* the loop. This happens because IEEE 754 floating-point addition is not associative: $(a + b) + c != a + (b + c)$. GCC must preserve the exact left-to-right evaluation order, preventing it from accumulating partial sums in vector registers.

Additionally, in v1_a:
- *Max-finding loop* (line 67-68): NOT vectorized ("unsupported use in stmt" - conditional max)
- *Softmax exp loop* (line 74-77): NOT vectorized ("statement clobbers memory" - `expf` call)

==== v1_b: True Vector Accumulation

Adding `-fassociative-math -fno-trapping-math -ffinite-math-only -fno-signed-zeros` transforms the dot product into truly vectorized code:

```asm
; GOOD: Vector FMA accumulation - no scalar ops inside loop
vmovups   zmm5, ZMMWORD PTR [rdi+rax] ; Load 16 floats from Q

; FMA: zmm0 += Q * K (all 16 lanes)
vfmadd231ps zmm0, zmm5, ZMMWORD PTR [rdx+rax]
add       rax, 64
cmp       rax, rcx
jne       .L_loop

; Horizontal reduction happens ONCE after the loop exits
vextractf32x8 ymm1, zmm0, 1               ; Extract upper 256 bits
vaddps    ymm0, ymm0, ymm1                ; Add upper to lower
vextractf128 xmm1, ymm0, 1                ; Extract upper 128 bits
vaddps    xmm0, xmm0, xmm1                ; Reduce to 128 bits
vmovshdup xmm1, xmm0                      ; Shuffle for final reduction
vaddps    xmm0, xmm1, xmm0                ; Continue reducing
vmovhlps  xmm1, xmm0, xmm0                ; Final shuffle
vaddss    xmm0, xmm0, xmm1                ; Final scalar add
```

The key difference: `vfmadd231ps` accumulates all 16 partial products in the `zmm0` vector register, and horizontal reduction to a scalar happens *once* after the loop completes. This is the ~3.5× speedup from v1_a to v1_b.

With these flags, the max-finding loop also vectorizes levereging `vmaxps zmm`, since `-ffinite-math-only` removes NaN handling requirements. 
The compiler now reports 5 vectorized loops instead of 4.

==== v1_c: Stride Padding for Alignment

Even with 64-byte aligned base pointers (via `posix_memalign`) and `head_dim=128`, the compiler cannot prove that every row access is aligned. The issue is that the *stride* between rows must also be a multiple of 64 bytes.

*v1_c* pads the storage stride to always be a multiple of `VEC_PADDING` (16 floats = 64 bytes):

```c
const size_t head_dim_padded = round_up_pow2(head_dim, VEC_PADDING);
size_t qkv_size = batch * n_heads * seq_len * head_dim_padded;
```

This ensures every row starts at a 64-byte boundary, avoiding cache-line splits during vectorized loads. 
On AMD Zen4, this provides ~30% improvement (note: modern CPUs handle `vmovups` efficiently when data is aligned).

==== v1_d: Vectorized exp computation

*v1_d* enables full `-ffast-math`, which finally allows vectorization of the `expf` loop (line 74-77). The compiler now reports:

```
kernels/single_thread/v1.cpp:74:42: optimized: loop vectorized using 64 byte vectors
kernels/single_thread/v1.cpp:25:6: note: vectorized 6 loops in function.
```

The generated assembly shows three variants of `expf` to handle different iteration counts:

```asm
call    _ZGVeN16v_expf    ; AVX-512: 16 floats at a time (main loop)
call    _ZGVdN8v_expf     ; AVX2: 8 floats (loop tail when 8-15 remain)
call    expf              ; Scalar (final cleanup for 1-7 elements)
```

These are GLIBC's SIMD math functions (`libmvec`). The three-tier approach handles: (1) the main vectorized loop processing 16 floats per iteration, (2) a secondary loop for remainders of 8-15 elements, and (3) scalar cleanup for the final 1-7 elements.

An alternatives approach would be to leverage *omp simd* or manually unroll the loops and include *Intel SVML intrinsics*, but `-ffast-math` streamlines the code much more.

*Remark on AVX-512 vector width:* On some architectures, GCC defaults to 256-bit vectors even when AVX-512 is available. 
During some testing I encounter thihs when working on CINECA's DCGP partition which has Intel Sapphire Rapids processors (e.g., Xeon Platinum 8480+). 
To force 512-bit vectors on these systems, I found adding `-mprefer-vector-width=512` to be sufficient. 

// ==== Vectorization Summary
//
// #table(
//   columns: (auto, auto, auto, auto, auto),
//   inset: 6pt,
//   align: center,
//   table.header([*Loop*], [*v1_a*], [*v1_b*], [*v1_c*], [*v1_d*]),
//   [Dot product], [Scalar reduce], [Vector FMA], [Aligned], [Aligned],
//   [Max-finding], [No], [Yes (vmaxps)], [Yes], [Yes],
//   [Softmax exp], [No], [No], [No], [Yes (vexp)],
//   [Normalization], [Yes], [Yes], [Yes], [Yes],
//   [Output accum], [Yes], [Yes], [Aligned], [Aligned],
//   table.hline(),
//   [*Total vectorized*], [4], [5], [5], [6],
// )

== Remaining Versions with small improvements (v2)

Versions v2 explore additional optimizations with diminishing returns on the single-threaded CPU backend:

- *v2* fuses the softmax computation phases (max-finding, exp, normalization) into tighter loops to reduce memory traffic. However, since `head_dim` is small (128) and fits in L1 cache, the measured improvement is negligible.

Moreover additional versions where tested but had no improvement:

- *v3* tried adding explicit `#pragma omp simd` hints to guide vectorization. With `-ffast-math` already enabled.

- *v4* implements Flash Attention-style online softmax, computing the running max and normalization factor in a single pass but requires computing exp multiple times. 

The key takeaway: once full vectorization is achieved (v1_d).
More on additional optimization techniques will be said in the following sections esepcially regarding GPU.

=== Final Summary of Results

The following figures showcases the actual results and timings.

#figure(
  image("../figures/benchmark_single.png", width: 95%),
  caption: [Single-threaded kernel benchmark results on AMD EPYC 9654 (Zen4). The plot shows execution time (lower is better) for each kernel version.],
) <fig:benchmark_single>
