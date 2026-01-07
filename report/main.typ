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
  academic-year: "2025-2026",
)

// Enable equation numbering and justify
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: set text(fill: blue)

#include "sections/introduction.typ"
#include "sections/background.typ"

// NOTE: I should add something on single_threaded desccribing the comparison pytorch versions 
#include "sections/single_thread.typ"

= Multi-Threaded CPU Implementation  


The benchmark has been runed like this on epyc partition of orfeo

The EPYC partition consists of the 8 nodes equipped with two AMD EPYC 7H12 cpus.

BENCH_BATCH=4 BENCH_HEADS=32 BENCH_SEQLEN=4096 BENCH_ITERS=10 


== First test Version (v0)

The first approach was a simple parallelization with collapose over ... Which is reasonable because things are separate etc ... The cores are placed close etc .. you can look at the sllurm script.
The result is quite reasonable already becuase of the well optimized code for v1 ... But with bigger input compared to the single threaded version it start beeing more reasonable to try to change the code to leverage the caceh more.

== Tiling over ... (v1)

One way to do so is the approach taken in v1 in this case the big chahnge was to ... moreover hitns to assume alligned ...

=== Final overview

The results with the benchmkar showcase ... A good scaling and ...

figure ...

Indeed the results are very good with comparable performance to the very optimized self attentino available from pytorch ... far exceeding the performance of the implementation you get when writing a pytorch code normally. 

= CUDA Implementation

I initially tried malloc Managed but for some reason even though I coundn't really see it clearly from the nsyight system the results were absolutly atrocious. 
Therefore i quickly switched to a direct allocation on the gpu with CudaMalloc and CudaMemcopy.

- v2 Added coalasced memory access

- v3 I was told to add this: --use_fast_math
Moreover I still have some uncoaleasced memory access so I have to think how to deal with that for key_pos which would make it much faster



[To be completed in next sections]

= Performance Analysis and Results

[To be completed in next sections]

= Conclusion and Future Work

[To be completed in next sections]

= References
#bibliography("refs.bib")
