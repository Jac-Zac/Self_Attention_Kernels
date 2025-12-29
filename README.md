# CMHSA (Causal Multi-Head Self-Attention)

Simple, learning-focused kernels:

- Single-thread CPU
- Multi-core CPU (OpenMP)
- CUDA (stub)

## Build

```bash
make single              # Single-threaded
make multi               # Multi-threaded (OpenMP)
make cuda                # CUDA (stub)
make single VERSION=v2   # Specific version
```

## Run

```bash
./cmhsa.out                          # Default parameters
./cmhsa.out --batch 4 --n_heads 8 --seq_len 512 --head_dim 64
./cmhsa.out --seed 42 --warmup 10 --iters 50   # Timing control
./cmhsa.out --threads 8              # Set OpenMP threads (multi only)
./cmhsa.out --validate-outdir DIR    # Export Q/K/V/out for validation
```

## Test

```bash
make test                # Validate all versions against PyTorch
```

Requires Python deps: `uv sync` or `pip install -r requirements.txt`

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
# Auto-detect plot type based on data
PYTHONPATH=python_src uv run python -m plot -i results/benchmark.csv
```

Outputs saved to `results/single_perf.png` or `results/strong_scaling.png`.

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
    __main__.py       # Auto-detect CLI
    single.py         # Bar plot for single-thread
    multi.py          # Strong scaling plot for multi-thread
    utils.py          # Shared plot utilities
  tests/              # Validation scripts
main.cpp              # Entry point
Makefile              # Build targets
```
