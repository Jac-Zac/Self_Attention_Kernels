= CUDA Implementation

This section presents the GPU implementation of causal multi-head self-attention using CUDA.
Unlike CPU implementations where we focused on SIMD vectorization and cache locality, GPU programming requires fundamentally different thinking: we must express algorithms in terms of thousands of concurrent threads, leverage the memory hierarchy effectively, and use specialized warp-level primitives for efficient communication between threads.

I develop multiple kernel versions: v0–v6 are implemented and v0–v3 are described below; v4 and later are experimental variants with vectorized and tiling improvements. The repository contains `v0.cu`, `v1.cu`, `v2.cu`, `v3.cu`, `v4.cu`, `v4.5.cu`, `v4.6.cu`, `v5.cu`, `v5.5.cu`, `v6.cu`, and a future experiment `future/v5_multi_query.cu`. Timings for v4+ and additional experiment variants are included in the results CSVs.
This implementation is FP32-only and does not use Tensor Cores (fp16/mixed-precision kernels), which provide higher throughput but add implementation complexity and precision considerations.
PyTorch's highly optimized SDPA kernels are further specialized for common head dimensions (64, 128), which partly explains the performance gap we observe.

== GPU Programming (some background)

Before diving into the implementations, I briefly review the GPU concepts most relevant to understanding the optimizations in this chapter.
For a more comprehensive treatment, I recommend the #link("https://modal.com/gpu-glossary")[Modal GPU Glossary] and NVIDIA's programming guides.

=== Warps and the SIMT Execution Model

The fundamental unit of execution on NVIDIA GPUs is the *warp*: a group of 32 threads that execute instructions in lockstep following the Single Instruction, Multiple Threads (SIMT) model.
All threads in a warp share a program counter and execute the same instruction simultaneously, though each operates on different data (similar to SIMD on CPUs, but with independent thread state).

Warps are scheduled onto *Streaming Multiprocessors (SMs)*, the GPU's compute units.
Each SM contains multiple *warp schedulers* that can rapidly switch between warps—context switches happen at the speed of a single clock cycle, over 1000× faster than CPU context switches.
This is possible because each thread has its own registers allocated from the SM's register file; no data movement is required to save or restore context.

This fast switching enables *latency hiding*: when one warp stalls on a memory access (which can take hundreds of cycles), the scheduler immediately switches to another eligible warp.
Consider a simple instruction sequence:

```nasm
LDG.E.SYS R1, [R0]        // memory load, ~400 cycles
IMUL R2, R1, 0xBEEF       // integer multiply, ~6 cycles
IADD R4, R2, 0xAFFE       // integer add, ~4 cycles
```

Executed sequentially, this takes ~410 cycles.
But if we have enough concurrent warps, the scheduler can keep the compute units busy while memory requests are in flight.
By Little's Law, with sufficient parallelism we can achieve one completed sequence per cycle on average.

The fraction of cycles on which a warp is issued an instruction is the *issue efficiency*, and the ratio of active warps to maximum possible warps is the *occupancy*.
High occupancy generally helps hide latency, though it is not always necessary for peak performance if computation is sufficiently intensive.

=== Memory Hierarchy and Coalescing

GPU memory forms a hierarchy with dramatically different bandwidths and latencies:

- *Registers*: Fastest storage, private to each thread. The register file is partitioned among all threads on an SM. Avoiding register spilling (overflow to local memory) is critical for performance.

- *Shared Memory / L1 Cache*: Fast, programmer-managed memory shared among threads in a *thread block* (also called Cooperative Thread Array or CTA in PTX terminology). Stored in the same physical SRAM as the L1 cache, with configurable partitioning. Typical access latency is ~20-30 cycles.

- *L2 Cache*: Shared across all SMs, caches global memory accesses.

- *Global Memory (HBM/GDDR)*: Large but slow. On data center GPUs like V100, High-Bandwidth Memory (HBM) provides ~900 GB/s bandwidth, but latency is still hundreds of cycles.

*Memory coalescing* is crucial for efficient global memory access.
When threads in a warp access memory, the hardware attempts to combine (coalesce) their requests into as few memory transactions as possible.
A single memory transaction can service 128 bytes—exactly enough for 32 threads to each load one 32-bit float.
For coalescing to work, threads must access contiguous addresses (though not necessarily in thread order).

```cpp
// GOOD: Coalesced - threads access consecutive addresses
float val = data[threadIdx.x];  // addresses 0x00, 0x04, 0x08, ...

// BAD: Strided - each thread hits different cache line
float val = data[threadIdx.x * 32];  // addresses 0x00, 0x80, 0x100, ...
```

Strided access patterns can result in 32× more memory transactions, severely limiting performance.

=== Warp-Level Primitives

CUDA provides *warp shuffle* intrinsics that allow threads within a warp to directly exchange register values without going through shared memory.
These are essential for efficient reductions:

- `__shfl_down_sync(mask, val, offset)`: Thread $i$ receives the value from thread $i + "offset"$
- `__shfl_xor_sync(mask, val, mask)`: Thread $i$ exchanges with thread $i xor "mask"$
- `__shfl_sync(mask, val, src)`: All threads receive the value from thread $"src"$ (broadcast)

The `sync` suffix and mask parameter ensure proper synchronization across the warp.
These operations complete in a single cycle and are fundamental to the optimizations in v1-v3.

=== Shared Memory Bank Conflicts

Shared memory is organized into 32 *banks*, with consecutive 4-byte words mapped to consecutive banks.
When multiple threads access different addresses in the same bank simultaneously, the accesses are serialized *bank conflict*.

```cpp
// NO conflict: consecutive addresses hit different banks
float val = smem[threadIdx.x];  // banks 0,1,2,...,31

// 32-way conflict: all threads hit bank 0
float val = smem[threadIdx.x * 32];  // all bank 0
```

Bank conflicts can increase shared memory latency by up to 32×.
In our implementations, we mostly avoid shared memory for the attention weights workspace, partly due to these concerns and partly due to occupancy limitations discussed later.
But this is something one has to be very careful of. Though one can read more about this #link("https://feldmann.nyc/blog/smem-microbenchmarks")[here]


== Benchmark Configuration

The CUDA benchmarks are run with the following configuration:

`batch=4`, `n_heads=32`, `seq_len=4096`, `head_dim=128`, `seed=1337`, `warmup=5`, `iters=25`.

Primary GPU: Cineca custom NVIDIA A100 (Ampere) accelerator. Profiling on a V100 was also performed as that was another GPU at my disposal.

Code: #link("https://github.com/Jac-Zac/Self_Attention_Kernels/tree/main/kernels/cuda")[github.com/Jac-Zac/Self_Attention_Kernels/tree/main/kernels/cuda].

Compilation uses `--use_fast_math`, which enables fast approximations for transcendental functions.
This generates Special Function Unit (SFU) instructions like `MUFU.EX2` for exponentials, providing significant speedups at the cost of slight precision loss (acceptable for attention weights).
Note that `cudaMalloc` automatically returns 256+ byte aligned pointers, satisfying alignment requirements for vectorized loads.

 == Naive Baseline (v0)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/cuda/v0.cu")[v0] is a straightforward port of the CPU algorithm to CUDA, assigning one thread per query position.

=== Grid and Thread Organization

The kernel uses a 3D grid mapping:
- *x dimension*: query positions (up to `seq_len`)
- *y dimension*: attention heads
- *z dimension*: batch (typically small, so z's 64-block limit is acceptable)

```cpp
int q = blockIdx.x * blockDim.x + threadIdx.x;  // query position
int h = blockIdx.y * blockDim.y + threadIdx.y;  // head
int b = blockIdx.z * blockDim.z + threadIdx.z;  // batch
```

Each thread independently computes attention for one query: computing all Q·K dot products, applying softmax, and accumulating the weighted sum of values.

=== Workspace Allocation

Each thread requires scratch space for `seq_len` attention weights (scores before/after softmax).
The workspace is indexed by a linearized global thread ID:

```cpp
int global_thread_id = block_id * threads_per_block + thread_id_in_block;
float *aw = attn_weights + global_thread_id * seq_len_padded;
```

This results in workspace size $cal(O)("total_threads" times "seq_len")$, which can be substantial.

=== Performance Characteristics

This baseline is extremely inefficient for several reasons:

1. *No parallelism within queries*: Each dot product (128 multiplications and additions) is computed sequentially by one thread, leaving most of the GPU idle.

2. *Poor memory access patterns*: Adjacent threads process different query positions, leading to strided access when loading Q vectors.

3. *Redundant memory traffic*: Each thread independently loads the same K and V rows, with no reuse across threads.

 The v0 kernel is approximately 46.4× slower than PyTorch's naive implementation on this workload (v0: 4.87710 s vs PyTorch naive: 0.10516 s); see `results/benchmark_gpu.csv`.

== Warp-Level Parallelism (v1)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/cuda/v1.cu")[v1] introduces the most important optimization: using an entire warp (32 threads) to collaboratively compute attention for a single query position.

=== Collaborative Dot Products

Instead of one thread computing the full dot product, all 32 warp threads participate:

```cpp
const int lane_id = threadIdx.x;  // 0-31 within warp

// Each thread handles dimensions: lane_id, lane_id+32, lane_id+64, ...
float dot_partial = 0.0f;
for (size_t d = lane_id; d < head_dim; d += warpSize) {
    dot_partial += Q[query_offset + d] * K[key_offset + d];
}

// Combine partial sums across warp
float dot_product = warp_reduce_sum(dot_partial);
```

This achieves two benefits:
1. *Parallelism*: The dot product computation is now 32× parallel
2. *Coalesced memory access*: Adjacent threads (lanes 0, 1, 2, ...) access consecutive memory addresses when loading Q and K vectors

=== Warp Reduction with Shuffle-Down

The partial sums must be combined into a single result.
The `warp_reduce_sum` function uses shuffle-down operations to perform a parallel reduction in $log_2(32) = 5$ steps:

```cpp
__inline__ __device__ float warp_reduce_sum(float val) {
    // Step 1: lanes 0-15 add from lanes 16-31
    // Step 2: lanes 0-7  add from lanes 8-15
    // Step 3: lanes 0-3  add from lanes 4-7
    // Step 4: lanes 0-1  add from lanes 2-3
    // Step 5: lane  0    adds from lane  1
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(WARP_MASK, val, offset);
    }
    return val;  // Result in lane 0 only
}
```

After reduction, only lane 0 holds the complete sum, so it writes the score to the workspace.
A `warp_broadcast` operation then distributes shared values (like the softmax normalization factor) to all lanes.

=== Triangular Workspace Optimization

Causal attention has a triangular structure: query at position $q$ only attends to keys $0, 1, ..., q$.
Rather than allocating rectangular workspace, v1 exploits this by allocating only the lower triangle:

```cpp
// Query q needs q+1 attention weights
// Total per (batch, head): sum_{i=0}^{seq_len-1} (i+1) = seq_len * (seq_len+1) / 2
const size_t workspace_per_bh = seq_len * (seq_len + 1) / 2;
const size_t triangular_offset = q * (q + 1) / 2;
```

This reduces workspace memory by approximately 50%.

=== Grid Configuration

The grid is reorganized to map one block per query position:

```cpp
dim3 threads_per_block(32, 1);  // One warp per block
dim3 number_of_blocks(seq_len, batch * n_heads);  // One block per query
```

This simplifies indexing: `blockIdx.x` directly gives the query position, and `blockIdx.y` encodes the batch-head pair.
  
V1 achieves approximately 8.4× speedup over v0, primarily from the warp-parallel dot products and coalesced memory access.

== XOR Reduction and Multi-Warp Blocks (v2)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/cuda/v2.cu")[v2] introduces two refinements: XOR-based reductions and multiple warps per block.

=== XOR vs Shuffle-Down Reduction

The shuffle-down reduction in v1 leaves the result only in lane 0, requiring a separate broadcast to distribute shared values.
XOR-based reduction naturally leaves the result in *all* lanes:

```cpp
__inline__ __device__ float warp_reduce_sum_xor(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(WARP_MASK, val, mask);
    }
    return val;  // Result in ALL lanes
}
```

The XOR pattern works as follows:
- `mask=16`: lanes 0-15 exchange with lanes 16-31
- `mask=8`: lanes 0-7,16-23 exchange with lanes 8-15,24-31
- ... and so on

After 5 iterations, every lane holds the complete sum.
This eliminates the need for explicit broadcast operations, simplifying the code flow.

=== Multi-Warp Blocks for Better Occupancy

V1 used only 32 threads per block (one warp).
V2 increases this to 256 threads (8 warps per block), with each warp handling a different query:

```cpp
#define WARPS_PER_BLOCK 8
dim3 threads_per_block(WARP_SIZE, WARPS_PER_BLOCK);  // 32 x 8 = 256

const int warp_id = threadIdx.y;  // Which warp in block (0-7)
const int lane_id = threadIdx.x;  // Lane within warp (0-31)
const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;  // Query position
```

This configuration provides several benefits:

1. *Better occupancy*: More threads per block means more warps can be scheduled on each SM, improving latency hiding.

2. *Reduced grid size*: The grid shrinks from `seq_len` blocks to `ceil(seq_len / 8)` blocks, reducing kernel launch overhead.

3. *Warp interleaving*: The scheduler can interleave execution of different warps within the same block, hiding memory latency more effectively.

V2 achieves approximately 1.37× additional speedup over v1.

== Online Softmax - Flash Attention Style (v3)

Version #link("https://github.com/Jac-Zac/Self_Attention_Kernels/blob/main/kernels/cuda/v3.cu")[v3] eliminates the attention weights workspace entirely using an *online softmax* algorithm, inspired by Flash Attention.

=== The Workspace Problem

In v1 and v2, we store all attention scores in global memory:
1. Compute all Q·K scores and write to workspace
2. Read scores, compute softmax, write back normalized weights
3. Read weights, accumulate weighted V sum

This requires $cal(O)("seq_len"^2)$ workspace per batch-head pair and multiple passes over the data.
For long sequences, workspace size becomes prohibitive, and the repeated global memory round-trips limit performance.

=== Online Softmax Algorithm

The key insight is that softmax can be computed *incrementally* as we iterate through keys, without storing all scores first.
We maintain running statistics that are updated with each new score:

```cpp
// State variables (in registers)
float running_max = -FLT_MAX;
float running_sum = 0.0f;
float output[head_dim] = {0};  // Accumulated in registers

for (int k = 0; k <= q; ++k) {
    // Compute Q·K score for key k
    float score = dot_product(Q[q], K[k]) * scale;
    
    // Online softmax update
    float new_max = fmaxf(running_max, score);
    float alpha = expf(running_max - new_max);   // Correction factor
    float weight = expf(score - new_max);        // New weight
    
    // Rescale previous accumulations and add new contribution
    running_sum = running_sum * alpha + weight;
    for (int d = 0; d < head_dim; d++) {
        output[d] = output[d] * alpha + weight * V[k][d];
    }
    
    running_max = new_max;
}

// Final normalization
for (int d = 0; d < head_dim; d++) {
    output[d] /= running_sum;
}
```

The mathematical justification is as follows.
Standard softmax computes:
$ "softmax"(s_i) = e^(s_i - m) / (sum_j e^(s_j - m)) $
where $m = max_j s_j$.

When we encounter a new score $s_k$ that changes the maximum from $m_"old"$ to $m_"new"$, we must rescale all previous exponentials:
$ e^(s_j - m_"new") = e^(s_j - m_"old") dot e^(m_"old" - m_"new") $

The factor $alpha = e^(m_"old" - m_"new")$ corrects all previously accumulated values.

=== Benefits

1. *Zero workspace*: The `cmhsa_get_workspace_size` function returns 0 for v3.

2. *Single pass*: We iterate through keys once, computing scores and accumulating outputs simultaneously.

3. *Register efficiency*: The output accumulator stays in registers (fast) rather than global memory (slow).

=== Profiling Analysis

Despite the algorithmic elegance, v3 provides measurable improvement over v2 but the gains are workload dependent.
Profiling with NVIDIA Nsight Compute reveals the bottleneck:

#figure(
  image("../figures/v3_ptx_slow.png", width: 100%),
  caption: [Nsight Compute profiling of v3 kernel showing warp stall analysis. The global memory load at line 240 (`ld.global.nc.f32`) accounts for 26% of stall cycles, indicating memory-bound behavior for the V matrix access pattern.],
) <fig:nsight_v3>

Key observations:

- *Stall distribution*: 26% of stalls occur at global memory loads for V vectors (line 240)
- *L1 cache hit rate*: Only ~37%, indicating poor temporal locality for K and V accesses
- *Compute vs memory bound*: The kernel is primarily stalled on memory, not arithmetic

The issue is that each query accesses K and V rows in sequence, but different queries (in different warps) access different subsets of rows.
Unlike the Q·K computation which benefits from warp-parallel coalescing, the sequential V accumulation has limited reuse opportunities.

 == v4 — Register-Resident Q and Output Accumulators

Version v4 (see `kernels/cuda/v4.cu`) builds on v3's online softmax but places both the query vector and the output accumulator entirely in registers. Key properties:

- Query in registers: each lane preloads up to 4 contiguous Q elements into per-lane registers (`q_r[]`) so the inner loop avoids reloading Q from memory.
- Register accumulator: the per-lane output (`out_accum[]`) is accumulated in registers and written once to global memory after normalization.
- Support: head_dim up to 128 using a per-lane chunking strategy (4 elements per lane × 32 lanes).
- Benefits: removes repeated Q loads and inner-loop read-modify-writes to global memory, improving arithmetic/memory balance and reducing global-memory traffic.
- Tradeoffs: increased register pressure (per-thread arrays) can limit occupancy on some architectures; performance depends on head_dim fitting the lane-chunking scheme.

  On the benchmark workload v4 runs in 0.228657669 s, improving on v3 but still trailing the vectorized variants (v4.5, v5, v6) in our additional experiments.

 == v4.5 — Float4 Vectorized Online Softmax

Version v4.5 (see `kernels/cuda/v4.5.cu`) further vectorizes the critical inner loop by operating on `float4` elements:

- Vectorized loads/stores: Q, K, V and the output use `float4` pointers where each lane handles one `float4`. This packs up to head_dim=128 with 4 floats per lane.
- Vectorized dot/accumulate: per-lane dot products and output updates are performed with `float4` components, reducing instruction count and memory traffic.
- Benefits: on workloads where head_dim is a multiple of 4 (and ≤128), float4 operations reduce memory pressure and improve throughput compared to scalar register-chunking.

In conclusion v4.5 runs in 0.142099548s. v4.5 is $approx 1.35×$ slower than PyTorch naive (0.105157305 s) and $approx 6.83×$ slower than PyTorch SDPA (0.020818206 s). 

== Performance Results

#figure(
  image("../figures/benchmark_gpu.png", width: 95%),
   caption: [CUDA kernel benchmark results. Left: speedup relative to PyTorch naive implementation. Right: speedup relative to PyTorch SDPA. 
    v4.5 is the best kernel without fancy optimization (0.14210 s), $approx 1.35×$ slower than PyTorch naive (0.105157305 s).]
) <fig:benchmark_gpu>

Detailed timing results are available in the @tab:benchmark_cuda.

=== Analysis

  The progression from v0 to v3 shows clear improvements in approach and wall-clock time. Measured speedups (baseline = PyTorch naive) are:
  - *v0 → v1*: v0 (4.8771 s) $->$ v1 (0.5860 s) $approx 8.32×$ reduction in time
  - *v1 → v2*: v1 (0.5860 s) $->$ v2 (0.4269 s) $approx 1.37×$ reduction in time
  - *v2 → v3*: v2 (0.4269 s) $->$ v3 (0.2546 s) $approx 1.68×$ reduction in time
  - *v3 → v4*: v3 (0.2546 s) $->$ v4 (0.2287 s) $approx 1.11×$ reduction in time
  - *v4 → v4.5*: v4 (0.2287 s) $->$ v4.5 (0.1421 s) $approx 1.61×$ reduction in time

=== The Gap with PyTorch SDPA

Several factors explain this performance gap:

1. *Tensor Cores*: PyTorch SDPA uses Tensor Cores for matrix multiplication, which provide 8-16× higher throughput than standard CUDA cores for half-precision operations. Our implementation uses only FP32 CUDA cores.

2. *Head dimension specialization*: PyTorch's kernels are specialized for common head dimensions (64, 128), enabling aggressive unrolling and register blocking. Our generic implementation cannot make these assumptions.

3. *Tiling and blocking*: Flash Attention tiles the computation to maximize shared memory reuse and minimize global memory traffic. Our v3 processes one key at a time without tiling.

4. *Memory access patterns*: Production kernels carefully orchestrate memory access to maximize coalescing and cache utilization across the entire computation, not just individual dot products.
