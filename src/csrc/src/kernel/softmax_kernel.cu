#include "softmax_kernel.cuh"
#include "utils.cuh"

#define BLOCK_SIZE 128
#define MAX_TILE_SIZE 32

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
    softmax_kernel(float* a, float* result, int N) {
  __shared__ float s_result[BLOCK_SIZE >> 5];
  __shared__ float s_inv_expsum;
  __shared__ float s_maxa;
  float ans = -INFINITY;
  float reg_result[MAX_TILE_SIZE];
  int base = blockIdx.x * N;
#pragma unroll
  for (int i = 0, pos = threadIdx.x; i < MAX_TILE_SIZE;
       ++i, pos += BLOCK_SIZE) {
    if (pos < N) {
      reg_result[i] = a[base + pos];
      ans = max(reg_result[i], ans);
    };
  }
  for (int i = 16; i >= 1; i >>= 1) {
    ans = max(ans, __shfl_xor_sync(0xffffffff, ans, i));
  }
  if ((threadIdx.x & 0x1f) == 0) {
    s_result[threadIdx.x >> 5] = ans;
  }
  __syncthreads();
  if (threadIdx.x < (BLOCK_SIZE >> 5)) {
    ans = s_result[threadIdx.x];
    for (int i = (BLOCK_SIZE >> 6); i >= 1; i >>= 1) {
      ans += max(ans, __shfl_xor_sync((1ll << (BLOCK_SIZE >> 5)) - 1, ans, i,
                                      (BLOCK_SIZE >> 5)));
    }
  }
  if (threadIdx.x == 0) {
    s_maxa = ans;
  }
  __syncthreads();
  float reg_maxa = s_maxa;
  ans = 0.f;
#pragma unroll
  for (int i = 0, pos = threadIdx.x; i < MAX_TILE_SIZE;
       ++i, pos += BLOCK_SIZE) {
    if (pos < N) {
      reg_result[i] = expf(reg_result[i] - reg_maxa);
      ans += reg_result[i];
    };
  }
  for (int i = 16; i >= 1; i >>= 1) {
    ans += __shfl_xor_sync(0xffffffff, ans, i);
  }
  if ((threadIdx.x & 0x1f) == 0) {
    s_result[threadIdx.x >> 5] = ans;
  }
  __syncthreads();
  if (threadIdx.x < (BLOCK_SIZE >> 5)) {
    ans = s_result[threadIdx.x];
    for (int i = (BLOCK_SIZE >> 6); i >= 1; i >>= 1) {
      ans += __shfl_xor_sync((1ll << (BLOCK_SIZE >> 5)) - 1, ans, i,
                             (BLOCK_SIZE >> 5));
    }
  }
  if (threadIdx.x == 0) {
    s_inv_expsum = 1 / ans;
  }
  __syncthreads();
  float reg_inv_expsum = s_inv_expsum;
#pragma unroll
  for (int i = 0, pos = threadIdx.x; i < MAX_TILE_SIZE;
       ++i, pos += BLOCK_SIZE) {
    if (pos < N) {
      float tmp = reg_result[i] * reg_inv_expsum;
      result[base + pos] = tmp;
    }
  }
}

#define EXPSUM_BLOCK_SIZE 256
#define EXPSUM_TILE_SIZE 32

__global__ void expsum_kernel(float* a, float* result, int N) {
  __shared__ float s_result[EXPSUM_BLOCK_SIZE >> 5];
  float ans = 0;
  for (int i = 0, pos = blockIdx.x * EXPSUM_BLOCK_SIZE * EXPSUM_TILE_SIZE +
                        threadIdx.x;
       i < EXPSUM_TILE_SIZE && pos < N; ++i, pos += EXPSUM_BLOCK_SIZE) {
    ans += expf(a[pos]);
  }
  for (int i = 16; i >= 1; i >>= 1) {
    ans += __shfl_xor_sync(0xffffffff, ans, i, 32);
  }
  if ((threadIdx.x & 0x1f) == 0) {
    s_result[threadIdx.x >> 5] = ans;
  }
  __syncthreads();
  if (threadIdx.x < (EXPSUM_BLOCK_SIZE >> 5)) {
    ans = s_result[threadIdx.x];
    for (int i = (EXPSUM_BLOCK_SIZE >> 6); i >= 1; i >>= 1) {
      ans += __shfl_xor_sync((1ll << (EXPSUM_BLOCK_SIZE >> 5)) - 1, ans, i,
                             (EXPSUM_BLOCK_SIZE >> 5));
    }
  } else {
    return;
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, ans);
  }
}

#define DIVIDE_BLOCK_SIZE 256

__global__ void divide_kernel(float* a, float* divider, float* result, int N) {
  int id = blockIdx.x * DIVIDE_BLOCK_SIZE + threadIdx.x;
  if (id < N) {
    float reg_a = a[id];
    float reg_divider = *divider;
    result[id] = expf(reg_a) / reg_divider;
  }
}

namespace {
float* expsum_result = nullptr;
bool inited = false;
}  // namespace

void softmax(float* a, float* result, int N) {
  if (N <= BLOCK_SIZE * MAX_TILE_SIZE) {
    softmax_kernel<<<1, BLOCK_SIZE>>>(a, result, N);
  } else {
    if (!inited) {
      global_exit_guard.Register([]() { softmaxDestroy(); });
      if (expsum_result != nullptr) {
        cudaFree(expsum_result);
      }
      cudaMallocAsync(&expsum_result, sizeof(float), 0);
      inited = true;
    }
    dim3 grid((N + EXPSUM_BLOCK_SIZE * EXPSUM_TILE_SIZE - 1) /
              (EXPSUM_BLOCK_SIZE * EXPSUM_TILE_SIZE));
    dim3 block(EXPSUM_BLOCK_SIZE);
    cudaMemsetAsync(expsum_result, 0, sizeof(float));
    expsum_kernel<<<grid, block>>>(a, expsum_result, N);
    grid = dim3((N + DIVIDE_BLOCK_SIZE - 1) / (DIVIDE_BLOCK_SIZE));
    block = dim3(DIVIDE_BLOCK_SIZE);
    divide_kernel<<<grid, block>>>(a, expsum_result, result, N);
  }
}

void softmax(float* a, float* result, int N, int M) {
  softmax_kernel<<<N, BLOCK_SIZE>>>(a, result, M);
}

void softmaxDestroy() {
  if (expsum_result != nullptr) {
    cudaFree(expsum_result);
    expsum_result = nullptr;
  }
  inited = false;
}
