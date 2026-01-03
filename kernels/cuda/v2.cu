#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// Standard Warp size for NVIDIA
#define WARP_SIZE 32

// Block-wide reduction for the Dot Product
// This handles cases where head_dim is 64, 128, etc.
__device__ float blockReduceSum(float val) {
  static __shared__ float shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  // 1. Warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  // 2. Final reduction across warps
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
  if (wid == 0) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // 2 floats of shared memory to broadcast scalars to the whole block
  __shared__ float s_max;
  __shared__ float s_sum;

  const int q = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int d = threadIdx.x;

  if (q >= dims.seq_len || d >= dims.head_dim)
    return;

  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = 1.0f / sqrtf((float)head_dim);

  const size_t workspace_per_bh = dims.seq_len * (dims.seq_len + 1) / 2;
  const size_t bh_workspace_offset = (b * dims.n_heads + h) * workspace_per_bh;
  const size_t triangular_offset = (size_t)q * (q + 1) / 2;
  float *RESTRICT aw = attn_weights + bh_workspace_offset + triangular_offset;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // Load query component once into register (SIMD load)
  float q_val = Q[query_offset + d];

  // Initialize block-wide max
  if (d == 0)
    s_max = -FLT_MAX;
  __syncthreads();

  // =====================================================================
  // Step 1: Dot Product & Max Search
  // =====================================================================
  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    size_t key_offset = bh_offset + key_pos * head_dim_pad;

    float prod = q_val * K[key_offset + d];
    float dot_product = blockReduceSum(prod);

    if (d == 0) {
      float score = dot_product * scale;
      aw[key_pos] = score;
      s_max = fmaxf(s_max, score); // Update shared max
    }
    __syncthreads(); // Ensure all threads see updated s_max
  }

  // =====================================================================
  // Step 2: Softmax Sum
  // =====================================================================
  if (d == 0)
    s_sum = 0.0f;
  __syncthreads();

  // Every thread contributes to the exp sum in a serial-per-thread way here
  // for the baseline, but we only need Thread 0 to compute it for the
  // broadcast.
  if (d == 0) {
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      float exp_val = expf(aw[key_pos] - s_max);
      aw[key_pos] = exp_val;
      s_sum += exp_val;
    }
  }
  __syncthreads(); // Ensure all threads see updated s_sum

  // =====================================================================
  // Step 3: Weighted Sum
  // =====================================================================
  float out_acc = 0.0f;
  const float inv_sum_exp = 1.0f / (s_sum + 1e-9f);

  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    size_t value_offset = bh_offset + key_pos * head_dim_pad;
    float weight = aw[key_pos] * inv_sum_exp;

    // SIMD style: thread 'd' handles dimension 'd'
    out_acc += weight * V[value_offset + d];
  }

  // Final coalesced write
  out[query_offset + d] = out_acc;
}

// ============================================================================
// Kernel Configuration
// ============================================================================

// Internal config struct - not exposed in header
typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  CudaConfig config;
  // SIMD philosophy: Threads = Head Dimension
  config.threads_per_block = dim3(dims.head_dim, 1, 1);

  // Blocks: x=queries, y=heads, z=batch
  config.number_of_blocks = dim3(dims.seq_len, dims.n_heads, dims.batch);

  config.total_threads = (size_t)config.number_of_blocks.x *
                         config.number_of_blocks.y * config.number_of_blocks.z *
                         config.threads_per_block.x;
  return config;
}

// ============================================================================
// Public API
// ============================================================================

size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  // Triangular workspace allocation leveraging causal mask pattern
  // Each (batch, head) needs: seq_len * (seq_len + 1) / 2 floats
  const size_t workspace_per_bh = dims.seq_len * (dims.seq_len + 1) / 2;
  return dims.batch * dims.n_heads * workspace_per_bh * sizeof(float);
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);

  VERBOSE_PRINT("CUDA Debug: Thread block (%d,%d,%d), Grid (%d,%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.threads_per_block.z, config.number_of_blocks.x,
                config.number_of_blocks.y, config.number_of_blocks.z);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "Block dimensions: (%d,%d,%d)\n",
            config.threads_per_block.x, config.threads_per_block.y,
            config.threads_per_block.z);
  }
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
