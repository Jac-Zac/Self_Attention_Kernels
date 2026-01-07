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
  caption: [Single-threaded kernel benchmark results on AMD EPYC 9654 (Zen4). Benchmark configuration: batch=1, heads=4, seq_len=4096, head_dim=128.],
) <tab:benchmark_single>

=== Multi-Threaded Strong Scaling Results

The following table presents detailed benchmark results for the multi-threaded CPU kernel implementations, showing strong scaling across different thread counts on the AMD EPYC 9654 (128 cores).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: none,
    table.hline(),
    table.header([*Threads*], [*v0 Time*], [*v1 Time*], [*v0 Speedup*], [*v1 Speedup*], [*PyTorch SDPA Time*]),
    table.hline(),
    [1], [23.6602], [20.6130], [1.00x], [1.00x], [9.5349],
    [2], [11.7860], [10.3062], [2.01x], [2.00x], [4.7776],
    [4], [8.0908], [5.1206], [2.92x], [4.02x], [2.4307],
    [8], [5.3945], [2.5808], [4.39x], [7.99x], [1.2295],
    [16], [3.1371], [1.3113], [7.54x], [15.72x], [0.6764],
    [32], [1.7142], [0.7207], [13.80x], [28.60x], [0.3682],
    [64], [0.9192], [0.4289], [25.73x], [48.06x], [0.2419],
    [128], [0.4997], [0.2198], [47.35x], [93.78x], [0.1499],
    table.hline(),
  ),
  caption: [Multi-threaded kernel strong scaling results on AMD EPYC 9654 (Zen4, 128 cores). Benchmark configuration: batch=4, heads=32, seq_len=4096, iters=10. Speedup is relative to v1 with 1 thread (20.6130s).],
) <tab:benchmark_strong_scaling>
