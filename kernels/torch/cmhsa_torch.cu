#include <ATen/cuda/CUDAContext.h>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define FULL_MASK 0xffffffff

// ============================================================
// Warp reduction
// ============================================================
__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(FULL_MASK, v, offset);
  return v;
}

// ============================================================
// CUDA kernel
// ============================================================
template <typename scalar_t>
__global__ void cmhsa_forward_kernel(const scalar_t *__restrict__ Q,
                                     const scalar_t *__restrict__ K,
                                     const scalar_t *__restrict__ V,
                                     scalar_t *__restrict__ O, int batch,
                                     int n_heads, int seq_len, int head_dim) {

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int bh = blockIdx.y;

  if (q >= seq_len)
    return;

  const int b = bh / n_heads;
  const int h = bh % n_heads;

  const size_t stride = (size_t)seq_len * head_dim;
  const size_t bh_offset = ((size_t)b * n_heads + h) * stride;

  const float scale = rsqrtf((float)head_dim);

  // ------------------------------------------------------------
  // Load Q into registers (scalar, no padding)
  // ------------------------------------------------------------
  constexpr int MAX_D_PER_LANE = 4; // 4 * 32 = 128
  float q_reg[MAX_D_PER_LANE];
  float out_acc[MAX_D_PER_LANE];

#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; ++i) {
    int d = lane + i * WARP_SIZE;
    q_reg[i] = (d < head_dim) ? (float)Q[bh_offset + q * head_dim + d] : 0.f;
    out_acc[i] = 0.f;
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.f;

  // ------------------------------------------------------------
  // Loop over keys (2 per iteration)
  // ------------------------------------------------------------
  int k = 0;
  for (; k + 1 <= q; k += 2) {

    float dot0 = 0.f;
    float dot1 = 0.f;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        float k0 = (float)K[bh_offset + (k + 0) * head_dim + d];
        float k1 = (float)K[bh_offset + (k + 1) * head_dim + d];
        dot0 += q_reg[i] * k0;
        dot1 += q_reg[i] * k1;
      }
    }

    dot0 = warp_reduce_sum(dot0) * scale;
    dot1 = warp_reduce_sum(dot1) * scale;

    // ---- score 0 ----
    {
      float new_max = fmaxf(running_max, dot0);
      float alpha = expf(running_max - new_max);
      float w = expf(dot0 - new_max);

      running_sum = running_sum * alpha + w;

#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; ++i) {
        int d = lane + i * WARP_SIZE;
        if (d < head_dim) {
          float v = (float)V[bh_offset + (k + 0) * head_dim + d];
          out_acc[i] = out_acc[i] * alpha + w * v;
        }
      }

      running_max = new_max;
    }

    // ---- score 1 ----
    {
      float new_max = fmaxf(running_max, dot1);
      float alpha = expf(running_max - new_max);
      float w = expf(dot1 - new_max);

      running_sum = running_sum * alpha + w;

#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; ++i) {
        int d = lane + i * WARP_SIZE;
        if (d < head_dim) {
          float v = (float)V[bh_offset + (k + 1) * head_dim + d];
          out_acc[i] = out_acc[i] * alpha + w * v;
        }
      }

      running_max = new_max;
    }
  }

  // ------------------------------------------------------------
  // Tail (if odd)
  // ------------------------------------------------------------
  if (k <= q) {
    float dot = 0.f;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        float kv = (float)K[bh_offset + k * head_dim + d];
        dot += q_reg[i] * kv;
      }
    }

    dot = warp_reduce_sum(dot) * scale;

    float new_max = fmaxf(running_max, dot);
    float alpha = expf(running_max - new_max);
    float w = expf(dot - new_max);

    running_sum = running_sum * alpha + w;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; ++i) {
      int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        float v = (float)V[bh_offset + k * head_dim + d];
        out_acc[i] = out_acc[i] * alpha + w * v;
      }
    }

    running_max = new_max;
  }

  // ------------------------------------------------------------
  // Write output
  // ------------------------------------------------------------
  float inv_sum = 1.f / running_sum;

#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; ++i) {
    int d = lane + i * WARP_SIZE;
    if (d < head_dim) {
      O[bh_offset + q * head_dim + d] = (scalar_t)(out_acc[i] * inv_sum);
    }
  }
}

// ============================================================
// PyTorch interface
// ============================================================
torch::Tensor cmhsa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
  TORCH_CHECK(Q.dtype() == K.dtype(), "dtype mismatch");
  TORCH_CHECK(Q.dtype() == V.dtype(), "dtype mismatch");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  const int batch = Q.size(0);
  const int n_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);

  TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");

  auto O = torch::zeros_like(Q);

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid((seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch * n_heads);

  at::cuda::CUDAGuard device_guard(Q.device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "cmhsa_forward_kernel", [&] {
        cmhsa_forward_kernel<scalar_t>
            <<<grid, block>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                              V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(),
                              batch, n_heads, seq_len, head_dim);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return O;
}

// ============================================================
// PyBind
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cmhsa_forward, "CMHSA forward (CUDA)");
}
