# CMHSA (Causal Multi-Head Self-Attention)

Simple, learning-focused kernels:

- Single-thread CPU
- Multi-core CPU (OpenMP)
- CUDA

## Build

```bash
make single              # Single-threaded
make multi               # Multi-threaded (OpenMP)
make cuda                # CUDA
make single VERSION=v2   # Specific version
```

## Run

```bash
./cmhsa.out                          # Default parameters (random Q,K,V)
./cmhsa.out --batch 4 --n_heads 8 --seq_len 512 --head_dim 64
./cmhsa.out --seed 42 --warmup 10 --iters 50   # Timing control
./cmhsa.out --threads 8              # Set OpenMP threads (multi only)
./cmhsa.out --input-dir DIR          # Load Q,K,V from directory
./cmhsa.out --validate-outdir DIR    # Export output tensor for validation
```

## Test

Tests validate kernel outputs against PyTorch's `scaled_dot_product_attention` using **real Q, K, V tensors extracted from GPT-2** attention layers.

```bash
make test                # Validate all versions against PyTorch
```

Requires Python deps: `uv sync`

## Benchmark

```bash
make benchmark                        # Default: single-threaded
make benchmark BENCH_BACKEND=multi BENCH_THREADS=8   # Multi-threaded
make benchmark BENCH_OUTPUT_FILE=results.csv         # Save to CSV
```

Override parameters as needed: `BENCH_BATCH`, `BENCH_HEADS`, `BENCH_SEQLEN`, `BENCH_HEADDIM`, `BENCH_ITERS`.

See `make help` for all options.

## Plotting

Generate plots locally after downloading benchmark CSV from cluster:

```bash
# Auto-detect plot type based on data (single/multi/cuda)
PYTHONPATH=python_src uv run python -m plot -i results/benchmark.csv
# Force specific plot type
PYTHONPATH=python_src uv run python -m plot --backend cuda -i results/cuda_benchmark.csv
```

Outputs saved to `results/single_perf.png`, `results/strong_scaling.png`, or `results/cuda_perf.png`.

## Project Structure

```bash
include/              # C++ headers
kernels/
  single_thread/      # CPU single-thread versions
  multi_thread/       # CPU OpenMP versions
  cuda/               # CUDA kernels
python_src/
  benchmark.py        # Benchmark runner
  utils.py            # Shared utilities
  plot/               # Plotting package
  tests/              # GPT-2 based validation tests
main.cpp              # CPU entry point
main.cu               # CUDA entry point
Makefile              # Build targets
```
