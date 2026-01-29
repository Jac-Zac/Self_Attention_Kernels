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
    [1], [23.6602], [20.6130], [1.00x], [1.15x], [9.5349],
    [2], [11.7860], [10.3062], [2.01x], [2.30x], [4.7776],
    [4], [8.0908], [5.1206], [2.92x], [4.62x], [2.4307],
    [8], [5.3945], [2.5808], [4.38x], [9.17x], [1.2295],
    [16], [3.1371], [1.3113], [7.54x], [18.06x], [0.6764],
    [32], [1.7142], [0.7207], [13.80x], [32.83x], [0.3682],
    [64], [0.9192], [0.4289], [25.73x], [55.19x], [0.2419],
    [128], [0.4997], [0.2198], [47.36x], [107.66x], [0.1499],
    table.hline(),
  ),
  caption: [Multi-threaded kernel strong scaling results on AMD EPYC 9654 (Zen4, 128 cores). Benchmark configuration: batch=4, heads=32, seq_len=4096, iters=10. Speedup values are relative to v0 with 1 thread (23.660236 s).],
) <tab:benchmark_strong_scaling>

=== CUDA Kernel Results

The following table presents detailed benchmark results for the CUDA kernel implementations compared against PyTorch GPU baselines. 

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: none,
    table.hline(),
    table.header([*Version*], [*Time (s)*], [*Speedup vs PyTorch naive*], [*Speedup vs PyTorch SDPA*]),
    table.hline(),
    [PyTorch naive], [0.105157305], [1.00×], [5.05×],
    [PyTorch SDPA], [0.020818206], [5.05×], [1.00×],
    table.hline(stroke: 0.5pt),
    [v0], [4.877098633], [0.022×], [0.004×],
    [v1], [0.585976929], [0.180×], [0.036×],
    [v2], [0.426904572], [0.246×], [0.049×],
    [v3], [0.254628952], [0.413×], [0.082×],
    [v4], [0.228657669], [0.460×], [0.091×],
    [v4.5], [0.142099548], [0.740×], [0.146×],
    table.hline(),
  ),
  caption: [CUDA kernel benchmark results (primary CSV: `results/benchmark_gpu.csv`). Benchmark configuration: batch=4, heads=32, seq_len=4096, head_dim=128. Speedups are computed as (baseline_time / version_time).],
) <tab:benchmark_cuda>

=== Additional GPU Runs (primary A100)

The repository also contains additional experimental runs on the primary A100 node. These include further vectorized and tiled variants (v4.6, v5.5, etc.). Times and speedups (baseline = PyTorch naive on A100) are shown below.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: none,
    table.hline(),
    table.header([*Version*], [*Time (s)*], [*Speedup vs PyTorch naive*], [*Speedup vs PyTorch SDPA*]),
    table.hline(),
    [PyTorch naive], [0.105157305], [1.00×], [5.05×],
    [PyTorch SDPA], [0.020818206], [5.05×], [1.00×],
    table.hline(stroke: 0.5pt),
    [v3], [0.254628952], [0.413×], [0.082×],
    [v4.5], [0.142099548], [0.740×], [0.146×],
    [v4.6], [0.096846359], [1.086×], [0.215×],
    [v5], [0.159500992], [0.659×], [0.130×],
    [v5.5], [0.134150757], [0.784×], [0.155×],
    [v6], [0.116500641], [0.903×], [0.179×],
    table.hline(),
  ),
  caption: [Additional CUDA runs on primary A100 (experimental variants). Speedups use the A100 PyTorch naive and SDPA baselines.],
) <tab:benchmark_cuda_additional>

#figure(
  image("../figures/benchmark_gpu_additional.png", width: 95%),
  caption: [Additional CUDA experiment plots (primary A100). These plots visualize the extended set of kernel variants including v4.6, v5.5 and v6.],
) <fig:benchmark_gpu_additional>

=== Additional GPU Runs (Orfeo V100)

Some of the best experimental kernels were also run on Orfeo's V100 node. The times below use the Orfeo PyTorch baselines measured on that machine (PyTorch naive = 0.18403 s, SDPA = 0.05404 s).

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: none,
    table.hline(),
    table.header([*Version*], [*Time (s)*], [*Speedup vs PyTorch naive (Orfeo)*], [*Speedup vs PyTorch SDPA (Orfeo)*]),
    table.hline(),
    [PyTorch naive], [0.184030977], [1.00×], [3.41×],
    [PyTorch SDPA], [0.054043281], [3.41×], [1.00×],
    table.hline(stroke: 0.5pt),
    [v4], [0.301160126], [0.612×], [0.179×],
    [v4.5], [0.201927185], [0.912×], [0.268×],
    [v4.6], [0.17616983], [1.045×], [0.307×],
    [v5], [0.275107758], [0.669×], [0.197×],
    [v5.5], [0.226585312], [0.812×], [0.239×],
    [v6], [0.191821213], [0.959×], [0.282×],
    table.hline(),
  ),
  caption: [Additional CUDA runs on Orfeo V100. These results show similar relative behavior: v4.6 and v6 are among the fastest in these runs on V100.],
) <tab:benchmark_orfeo_additional>

#figure(
  image("../figures/benchmark_gpu_orfeo_additional.png", width: 95%),
  caption: [Orfeo V100 additional experiment plots (detailed).],
) <fig:benchmark_gpu_orfeo_additional>
