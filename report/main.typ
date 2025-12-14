#import "@preview/red-agora:0.1.2": project

// HACK: To now show table of figures and table of tables
#show outline: it => {
  let is-figure = it.target == figure.where(kind: image)
  let is-table = it.target == figure.where(kind: table)
  
  if is-figure or is-table {
    none
  } else {
    it
  }
}

#show: project.with(
  title: "Tutorial: Implementing Efficient Causal Multi-Head Self-Attention for CPU and GPU",
  subtitle: "A Performance-Oriented Study of Self-Attention Kernels Across Computing Architectures",
  authors: (
    "Jacopo Zacchigna",
  ),
  school-logo: [],
  company-logo: [],
  mentors: (
    "Pr. John Smith (Internal)",
  ),
  branch: "High Performance Computing",
  academic-year: "2024-2025",
)

// Enable equation numbering and justify
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: set text(fill: blue)

#bibliography("refs.bib")

#include "sections/introduction.typ"
#include "sections/background.typ"

= Single-Threaded CPU Implementation

[To be completed in next sections]

=== Notable result for version 1 (v1)


Although the compiler appears to vectorize the naive solution based on its outputs, the generated code is not fully vectorized. 
One clear example can be seen in the self-attention operation.

```c
// V1.cpp approx line 50
dot_product += Q[query_offset + d] * K[key_offset + d];
```

Generates the following assembly:

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


Instead of keeping YMM registers as partial sums and reducing them afterward, the compiler immediately reduces to scalar operations inside the loop, providing no speedup. 
This occurs because we use floating-point values, and GCC prioritizes *IEEE 754 compliance* without additional flags. 
Since addition is not associative $(a + b) + c != a + (b + c)$ the compiler must revert to scalar instructions.

Another case where this occurs is ...


= Multi-Threaded CPU Implementation  

[To be completed in next sections]

= CUDA Implementation

[To be completed in next sections]

= Performance Analysis and Results

[To be completed in next sections]

= Conclusion and Future Work

[To be completed in next sections]

= References
// #bibliography("refs.bib")
