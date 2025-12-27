# CMHSA (Causal Multi-Head Self-Attention)

Simple, learning-focused kernels:

- Single-thread CPU
- Multi-core CPU (OpenMP)
- CUDA (stub)

## Build

- Single, Multi, Cuda: `make single`, `make multi`, `make cuda`
- Choose version: `make single VERSION=v0`

## Run

> Causal attention is always enabled.

- `./cmhsa.out` runs with defaults.
- Flags (errors on unknown/missing values):
  - `--validate-outdir DIR` (writes Q/K/V/out and meta for Python)
  - `--batch N` `--n_heads N` `--seq_len N` `--head_dim N`
  - `--seed N` `--warmup N` `--iters N`

## Test

- `make test` runs the lightweight validation program that uses the same parser.
- Python deps: `source .venv/bin/activate` then `pip install -r requirements.txt` (or `uv sync`).
- Validation script: `python_src/tests/validate_with_torch.py`
- CUDA tests not implemented yet.

## Benchmark

- Easiest: `make benchmark` builds and runs preset sizes, printing timing lines for each version.
- Timings are reported in milliseconds via `timing.h`.

### Scaling Analysis

For multi-threaded scaling analysis:

1. Run benchmarks at different thread counts, saving JSON output
2. Merge results: `python python_src/merge_results.py results/*.json -o results/combined.json`
3. Generate plots: `python python_src/plotting/scaling_plot.py -i results/combined.json`

The SLURM script `slurm_scripts/orfeo/benchmark` automates this workflow.

## Project Structure

- `include/`: headers (`cmhsa_forward.h`, `macros.hpp`, `timing.h`, `utils.hpp`)
- `kernels/`: implementations
  - `single_thread/`: CPU single-thread versions
  - `multi_thread/`: CPU OpenMP versions
  - `cuda/`: CUDA stubs/kernels
- `python_src/`: Python scripts
  - `utils.py`: shared utilities
  - `benchmark_all.py`: benchmark runner with PyTorch comparison
  - `merge_results.py`: merge multiple benchmark JSONs
  - `tests/validate_with_torch.py`: validation against PyTorch
  - `plotting/scaling_plot.py`: generate scaling plots
- `main.cpp`: runnable example using the kernels
- `Makefile`: build targets (single, multi, cuda)
- `report/`: Typst report
- `slurm_scripts/`: SLURM job scripts for HPC clusters
