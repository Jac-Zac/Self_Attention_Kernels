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
  - `--validate-outdir DIR`
  - `--batch N` `--n_heads N` `--seq_len N` `--head_dim N`
  - `--seed N`
- Causal is always enabled.

## Test

- `make test` runs the lightweight validation program that uses the same parser.
- Artifacts written under `python_test/`: `q.bin`, `k.bin`, `v.bin`, `out.bin`, `meta.json`.
- CUDA tests not implemented yet.

## Project Structure

- `include/`: headers (`cmhsa_forward.h`, `macros.hpp`, `timing.h`, `utils.hpp`)
- `kernels/`: implementations
  - `single_thread/`: CPU single-thread versions
  - `multi_core_cpu/`: CPU OpenMP versions
  - `cuda/`: CUDA stubs/kernels
- `tests/`: C++ validation/test programs
- `python_tests/`: Torch validation script which does checks against torch version
- `main.cpp`: runnable example using the kernels
- `Makefile`: build targets (single, multi, cuda)
- `report/`: Typst report
