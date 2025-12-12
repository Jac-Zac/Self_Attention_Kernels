#import "@preview/red-agora:0.1.2": project

// HACK: To now show table of figures and table of tables
#show outline: it => {
  // Check if this outline is for figures or tables
  let is-figure = it.target == figure.where(kind: image)
  let is-table = it.target == figure.where(kind: table)
  
  if is-figure or is-table {
    none  // Don't show this outline
  } else {
    it  // Show other outlines (like TOC)
  }
}

#show: project.with(
  title: "Tutorial: Implementing Efficient Causal Multi-Head Self-Attention for CPU and GPU",
  subtitle: "A Performance-Oriented Study of Self-Attention Kernels Across Computing Architectures",
  authors: (
    "Jacopo Zacchigna",
  ),
  school-logo: [], // Replace with image("images/log.svg") to remove the school logo
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

= Introduction


Transformer architectures have had a drastic impact on many fields, foremost among them natural language processing @vaswani2017attention. At the heart of these models lies the attention mechanism, which enables dynamic, context-dependent weighting of input sequences. However, this mechanism is also the main computational bottleneck, particularly due to its quadratic scaling with respect to the context length. This makes self-attention, especially for long sequences, one of the most interesting problems to optimize.

This project dives into different efficient ways to optimize the *causal multi-head self-attention* (CMHA) operation, which is the variant used in autoregressive language models like GPT @radford2018improving @radford2019language @brown2020language. I develop and benchmark three different sets of progressively optimized implementations:

1. *Single-threaded CPU*: A baseline implementation with various improvements leveraging different vectorization techniques.
2. *Multi-threaded CPU*: An OpenMP-parallelized version that exploits modern multi-core architectures.
3. *GPU (CUDA)*: Some massively parallel implementation leveraging GPU memory hierarchies

// TODO: Conclude this
For educational purposes: each implementation builds upon the previous one, explicitly addressing performance bottlenecks and leveraging as much of the hardwar as possible. I validate correctness against PyTorch's reference implementation and analyze performance characteristics on ... (I have to decide where to run it)

== Scope of the implementation

The focus of this project is the implementation of the *causal scaled dot-product attention kernel*. 
The implementation operates directly on pre-projected query, key, and value tensors $bold(Q)$, $bold(K)$, and $bold(V)$, which are assumed to be already partitioned per attention head and laid out contiguously in memory.
As such, the linear input projections $bold(X) bold(W)_q$, $bold(X) bold(W)_k$, and $bold(X) bold(W)_v$ are not included.
Furthermore, this implementation computes attention independently for each head. Head concatenation, the output projection $bold(W)_o$, and residual connections are intentionally omitted, as they can be performed sepratly from the attention kernel I will focus on.

== Motivation and Scope

The quadratic complexity of self-attention ($cal(O)(T^2 d)$ for sequence length $T$ and model dimension $d$) presents fundamental challenges:

- *Memory bandwidth limitations*: Naive implementations repeatedly transfer large matrices between main memory and compute units
- *Materialization costs*: The full attention matrix $bold(Q) bold(K)^top in bb(R)^(T times T)$ may exceed fast cache/SRAM capacity

Recent work like FlashAttention @dao2022flashattention demonstrates that careful algorithm-hardware co-design can achieve significant speedups (2-4× for training, 10-20× for long-context inference). Though I will start by simply implementing versions from my previous knoweldge and then building up to more and more sophisticated tricks.

= Background and Theory

This section establishes the mathematical foundations of causal multi-head self-attention, used in modern Transformers.

== Scaled Dot-Product Attention and Self-Attention

The fundamental attention mechanism computes a weighted sum of values based on query-key similarities. Given query matrix $bold(Q) in bb(R)^(T times d_k)$, key matrix $bold(K) in bb(R)^(T times d_k)$, and value matrix $bold(V) in bb(R)^(T times d_v)$, where $T$ is sequence length, the attention output is:

$ bold(A) = "softmax"((bold(Q) bold(K)^top)/sqrt(d_k)) bold(V) $ <eq:attention>

The computation proceeds in three stages: (1) *similarity computation* $bold(S) = bold(Q) bold(K)^top$ computes pairwise token relationships, (2) *normalization* via $"softmax"(bold(S) "/" sqrt(d_k))$ converts similarities to probability distributions, and (3) *aggregation* computes weighted combinations of values. The scaling factor $1\/sqrt(d_k)$ prevents dot products from growing as $cal(O)(sqrt(d_k))$ in magnitude, which would push softmax into regions with vanishing gradients @vaswani2017attention.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 0em,
    image("figures/scaled_dot-product_attention.png", height: 25%),
    image("figures/multi-head-attention.png")
  ),

  caption: [*Left*: Scaled dot-product attention computes query-key similarities, applies causal masking, normalizes via softmax, and weights the values. *Right*: Multi-head attention architecture projects input into $h$ parallel subspaces, each computing independent attention before concatenation.]
) <fig:attention-mechanisms>

In *self-attention*, all three matrices derive from the same input sequence $bold(X) in bb(R)^(T times d_"model")$ via learned linear projections:

$ bold(Q) = bold(X) bold(W)_q, quad bold(K) = bold(X) bold(W)_k, quad bold(V) = bold(X) bold(W)_v $ <eq:projections>

where $bold(W)_q, bold(W)_k in bb(R)^(d_"model" times d_k)$ and $bold(W)_v in bb(R)^(d_"model" times d_v)$ are learned parameters. This allows each token to determine what to look for (query), what it offers (key), and what information to provide (value). Though this is not the focus of my implementation.

=== The Residual Stream 

Transformer blocks operate on a *residual stream* @elhage2021mathematical—a continuous representation that flows through the network, modified incrementally by each layer:

$ bold(X)_"out" = bold(X) + "Attention"(bold(X)) bold(W)_o $ <eq:residual>

where $bold(W)_o in bb(R)^(d_v times d_"model")$ projects attention outputs back to the model dimension. This additive structure enables gradient flow through deep networks, and is the core of the Transformer @elhage2021mathematical.

== Multi-Head Attention

Single attention heads struggle to capture diverse patterns simultaneously on the same layer. Multi-head attention (MHA) parallelizes computation across $h$ independent heads, each operating in a lower-dimensional subspace:

$ "head"_i = "Attention"(bold(X) bold(W)_q^((i)), bold(X) bold(W)_k^((i)), bold(X) bold(W)_v^((i))) $ <eq:single-head>

where each head has dimensionality $d_k = d_v = d_"model" \/ h$. The outputs are concatenated and projected:

$ op("MHA")(bold(X)) = op("concat")("head"_1, dots, "head"_h) bold(W)_o $ <eq:mha>

In this work, I isolate and optimize the per-head attention computation, thus not focusing on head concatenation and output projection.
This design provides a sort of specialization which allows different heads to learn complementary patterns. Note that despite using $h$ heads, the total dimensionality remains $d_"model"$ (since $h dot.op d_"model"\/h = d_"model"$), making parameter count comparable to a single large head.

== Causal Masking for Autoregressive Models

In autoregressive language modeling, the model must not access future tokens during training. This constraint is enforced via a *causal mask* that zeros out attention to future positions:

$ bold(A)_"causal" = "softmax"((bold(Q) bold(K)^top + bold(M))/sqrt(d_k)) bold(V) $ <eq:causal>

where the mask is:

$ bold(M)_(i j) = cases(
  0 & "if" i >= j "  (position" i "can attend to" j")",
  -infinity & "if" i < j "  (position" i "cannot attend to" j")"
) $ <eq:mask>

The $-infinity$ values become zero after softmax, effectively removing future positions from the attention distribution. 
This creates a *lower triangular*, indeed attention pattern at position $i$ only attends to positions $<= i$.

== Computational Complexity and Implementation Challenges

The attention mechanism has $cal(O)(T^2 d)$ complexity, dominated by the query-key multiplication ($cal(O)(T^2 d_k)$) and attention-value multiplication ($cal(O)(T^2 d_v)$). The quadratic scaling with sequence length becomes a bottleneck for long contexts.

*Softmax numerical stability*: Standard softmax $"softmax"(x_i) = e^(x_i) / (sum_j e^(x_j))$ is numerically unstable due to exponential overflow for large positive values and underflow for large negative values. 
The stable version:
$
"softmax"(x_i) = (e^(x_i - x_"max")) / (sum_j e^(x_j - x_"max"))
$
subtracts the maximum before exponentiation, keeping all exponents non-positive to prevent overflow while maintaining mathematical equivalence @goodfellow2016deep.

= Single-Threaded CPU Implementation

[To be completed in next sections]

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
