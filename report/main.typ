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
#include "sections/single_thread.typ"

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
