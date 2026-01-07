= Appendix

== Benchmark Results

=== Single-Threaded CPU Kernel Results

The following table presents detailed benchmark results for the single-threaded CPU kernel implementations compared against PyTorch baselines.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: none,
    table.hline(),
    table.header([*Version*], [*Threads*], [*Time (s)*], [*Speedup vs Naive*], [*Speedup vs SDPA*]),
    table.hline(),
    [PyTorch naive], [1], [1.0483], [1.00x], [5.82x],
    [PyTorch SDPA], [1], [0.1802], [0.17x], [1.00x],
    table.hline(stroke: 0.5pt),
    [v0], [1], [11.1723], [0.09x], [0.02x],
    [v1_a], [1], [2.2168], [0.47x], [0.08x],
    [v1_b], [1], [0.6039], [1.74x], [0.30x],
    [v1_c], [1], [0.4531], [2.31x], [0.40x],
    [v1_d], [1], [0.3711], [2.82x], [0.49x],
    [v2], [1], [0.3706], [2.83x], [0.49x],
    table.hline(),
  ),
  caption: [Single-threaded kernel benchmark results on AMD EPYC 9654 (Zen4). Benchmark configuration: batch=4, heads=8, seq_len=512, head_dim=64.],
) <tab:benchmark_single>
