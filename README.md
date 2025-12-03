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

- `./cmhsa [N]` (default `N=1024`)
- Runtime prints: `backend`, `version`, `n`, timing in seconds

## Test

- `make test` (runs single+multi for all versions)
- CUDA tests not implemented yet
