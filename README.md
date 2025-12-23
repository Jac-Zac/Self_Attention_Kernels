# CMHSA (Causal Multi-Head Self-Attention)

Simple, learning-focused kernels:

- Single-thread CPU
- Multi-core CPU (OpenMP)
- CUDA (stub)

## Build

- Single: `make single`
- Multi: `make multi`
- CUDA: `make cuda`
- Choose version: `make single VERSION=v0`
- Choose output name/extension: `make single EXEC=cmhsa.out`

## Run

- `./cmhsa` runs with defaults.
- Flags (errors on unknown/missing values):
  - `--validate-outdir DIR` (writes Q/K/V/out and meta for Python)
  - `--batch N` `--n_heads N` `--seq_len N` `--head_dim N`
  - `--seed N` `--warmup N` `--iters N`
- Causal is always enabled.

## Test

- `make test` runs the lightweight validation program that uses the same parser.
- Python deps: `source .venv/bin/activate` then `pip install -r requirements.txt` (or `uv sync`).
- Artifacts written under `python_tests/`: `q.bin`, `k.bin`, `v.bin`, `out.bin`, `meta.json`.
- CUDA tests not implemented yet.

## Benchmark

- Easiest: `make benchmark` builds and runs preset sizes, printing timing lines for each version.
- Threads: pass `--threads N` to the binary (benchmark script forwards it) and/or set `OMP_NUM_THREADS`.
- Timings are reported in milliseconds via `timing.h`.

## Project Structure

- `include/`: headers (`cmhsa_forward.h`, `macros.hpp`, `timing.h`, `utils.hpp`)
- `kernels/`: implementations
  - `single_thread/`: CPU single-thread versions
  - `multi_core_cpu/`: CPU OpenMP versions
  - `cuda/`: CUDA stubs/kernels
- `python_tests/`: Torch validation script which does checks against torch version
- `main.cpp`: runnable example using the kernels
- `Makefile`: build targets (single, multi, cuda)
- `report/`: Typst report
