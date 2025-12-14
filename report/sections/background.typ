= Background and Theory

This section explains the mathematical foundations of causal multi-head self-attention, used in modern Transformers.

== Scaled Dot-Product Attention and Self-Attention

The fundamental computation in the attention mechanism is a weighted sum of values based on query-key similarities. 
Given query matrix $bold(Q) in bb(R)^(T times d_k)$, key matrix $bold(K) in bb(R)^(T times d_k)$, and value matrix $bold(V) in bb(R)^(T times d_v)$, where $T$ is sequence length, the attention output is:

$ bold(A) = "softmax"((bold(Q) bold(K)^top)/sqrt(d_k)) bold(V) $ <eq:attention>

The computation proceeds in three stages: (1) *similarity computation* $bold(S) = bold(Q) bold(K)^top$ computes pairwise token relationships, (2) *normalization* via $"softmax"(bold(S) "/" sqrt(d_k))$ converts similarities to probability distributions, and (3) *aggregation* computes weighted combinations of values. The scaling factor $1\/sqrt(d_k)$ prevents dot products from growing as $cal(O)(sqrt(d_k))$ in magnitude @vaswani2017attention.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 0em,
    image("../figures/scaled_dot-product_attention.png", height: 25%),
    image("../figures/multi-head-attention.png")
  ),

  caption: [*Left*: Scaled dot-product attention computes query-key similarities, applies causal masking, normalizes via softmax, and weights the values. *Right*: Multi-head attention architecture projects input into $h$ parallel subspaces, each computing independent attention before concatenation.]
) <fig:attention-mechanisms>

In *self-attention*, all three matrices derive from the same input sequence $bold(X) in bb(R)^(T times d_"model")$ via learned linear projections:

$ bold(Q) = bold(X) bold(W)_q, quad bold(K) = bold(X) bold(W)_k, quad bold(V) = bold(X) bold(W)_v $ <eq:projections>

where $bold(W)_q, bold(W)_k in bb(R)^(d_"model" times d_k)$ and $bold(W)_v in bb(R)^(d_"model" times d_v)$ are learned parameters. 
This allows each token to determine what to look for (query), what it offers (key), and what information to provide (value). Though this is not the focus of my implementation.

=== The Residual Stream 

Transformer blocks operate on a *residual stream* @elhage2021mathematical—a continuous representation that flows through the network, modified incrementally by each layer:

$ bold(X)_"out" = bold(X) + "Attention"(bold(X)) bold(W)_o $ <eq:residual>

where $bold(W)_o in bb(R)^(d_v times d_"model")$ projects attention outputs back to the model dimension. 
This additive structure enables gradient flow through deep networks, and is the core of the Transformer @elhage2021mathematical.

== Multi-Head Attention

Single attention heads struggle to capture diverse patterns simultaneously on the same layer. 
Multi-head attention (MHA) parallelizes computation across $h$ independent heads, each operating in a lower-dimensional subspace:

$ "head"_i = "Attention"(bold(X) bold(W)_q^((i)), bold(X) bold(W)_k^((i)), bold(X) bold(W)_v^((i))) $ <eq:single-head>

where each head has dimensionality $d_k = d_v = d_"model" \/ h$. The outputs are concatenated and projected:

$ op("MHA")(bold(X)) = op("concat")("head"_1, dots, "head"_h) bold(W)_o $ <eq:mha>

In this work, I isolate and optimize the per-head attention computation, thus not focusing on head concatenation and output projection.
This design provides a sort of specialization which allows different heads to learn complementary patterns. 
Note that despite using $h$ heads, the total dimensionality remains $d_"model"$ (since $h dot.op d_"model"\/h = d_"model"$), making parameter count comparable to a single large head.

== Causal Masking for Autoregressive Models

In autoregressive language modeling, the model must not access future tokens during training. 
This constraint is enforced via a *causal mask* that zeros out attention to future positions:

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
