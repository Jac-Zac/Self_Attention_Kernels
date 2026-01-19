#ifndef PTX_UTILS_CUH
#define PTX_UTILS_CUH

// ============================================================================
// PTX Utilities for Async Memory Operations (sm_80+)
// ============================================================================
// This header provides inline PTX wrappers for asynchronous GMEM->SMEM copies.
//
// Key benefits of cp.async over regular loads:
// 1. Bypasses L1 cache (.cg = cache global = L2 only)
// 2. Direct GMEM->SMEM path (no register file intermediate)
// 3. Enables overlapped load/compute via commit groups
// 4. Reduces register pressure for large transfers
//
// Usage pattern for double buffering:
//   1. cp_async<16>() x N times to fill buffer A
//   2. cp_async_commit_group() - creates commit group 0
//   3. cp_async<16>() x N times to fill buffer B
//   4. cp_async_commit_group() - creates commit group 1
//   5. cp_async_wait<1>() - wait for group 0 (1 group outstanding)
//   6. Process buffer A while buffer B loads continue
//   7. Repeat...
//
// Requirements:
// - sm_80 or higher (Ampere+)
// - SMEM destination must be properly aligned (16 bytes for cp_async<16>)
// - Compile with: nvcc -arch=sm_80 (or higher)
//
// ============================================================================

#include <cuda_runtime.h>

// Async copy 16 bytes from global memory to shared memory
// Uses .cg (cache global) modifier to bypass L1 cache
// smem_ptr: pointer to shared memory destination (must be 16-byte aligned)
// gmem_ptr: pointer to global memory source (must be 16-byte aligned)
__device__ __forceinline__ void cp_async_16(void *smem_ptr,
                                            const void *gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  // cp.async.cg.shared.global [dst], [src], 16;
  // .cg = cache at global level (L2 only, bypass L1)
  // .shared.global = shared memory destination, global memory source
  // 16 = number of bytes to copy
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                   static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr))),
               "l"(gmem_ptr));
#else
  // Fallback for older architectures: synchronous copy via registers
  *reinterpret_cast<float4 *>(smem_ptr) =
      *reinterpret_cast<const float4 *>(gmem_ptr);
#endif
}

// Async copy 16 bytes with predicate (conditional copy)
// If pred is false, the copy is skipped but still counts toward the group
// Useful for bounds checking without divergent branches
__device__ __forceinline__ void cp_async_16_pred(void *smem_ptr,
                                                 const void *gmem_ptr,
                                                 bool pred) {
#if __CUDA_ARCH__ >= 800
  // cp.async.cg.shared.global [dst], [src], 16, p;
  // The predicate version: if p is false, no copy happens but the
  // async operation still commits (avoids group size mismatch)
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
      "}\n" ::"r"(static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr))),
      "l"(gmem_ptr), "r"((int)pred));
#else
  if (pred) {
    *reinterpret_cast<float4 *>(smem_ptr) =
        *reinterpret_cast<const float4 *>(gmem_ptr);
  }
#endif
}

// Async copy 16 bytes, zeroing destination if out of bounds
// If pred is false, writes zeros to smem instead of copying
// Useful for handling partial tiles at sequence boundaries
__device__ __forceinline__ void cp_async_16_zfill(void *smem_ptr,
                                                  const void *gmem_ptr,
                                                  bool pred) {
#if __CUDA_ARCH__ >= 800
  // With src-size=0, the destination is zero-filled
  // This variant copies 16 bytes if pred is true, else fills with zeros
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
      "  @!p cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
      "}\n" ::"r"(static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr))),
      "l"(gmem_ptr), "r"((int)pred));
#else
  if (pred) {
    *reinterpret_cast<float4 *>(smem_ptr) =
        *reinterpret_cast<const float4 *>(gmem_ptr);
  } else {
    *reinterpret_cast<float4 *>(smem_ptr) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
#endif
}

// Commit the current group of async copies
// All cp.async calls since the last commit (or kernel start) form a group
// Groups complete in order, enabling pipelined double/triple buffering
__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait for async copy groups to complete
// N = number of groups that may still be in flight after this call
// wait<0>: wait for ALL groups to complete
// wait<1>: wait until at most 1 group is still in flight
// wait<2>: wait until at most 2 groups are still in flight
//
// For double buffering: use wait<1> to ensure previous buffer is ready
// while current buffer is still loading
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Wait for ALL async copies to complete
// Equivalent to cp_async_wait_group<0>()
// Use this before accessing any data from the final async copy group
__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// Fence to ensure visibility of async copies within the CTA
// Call this after wait_group/wait_all before reading the data
// This is a memory fence, not a thread sync (use __syncthreads() for that)
__device__ __forceinline__ void cp_async_fence() {
#if __CUDA_ARCH__ >= 800
  asm volatile("fence.proxy.async.shared::cta;\n" ::);
#endif
}

#endif // PTX_UTILS_CUH
