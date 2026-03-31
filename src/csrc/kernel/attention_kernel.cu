#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cublas_handle.cuh"
#include "softmax_kernel.cuh"
#include "utils.cuh"

__global__ void apply_causal_mask(float* a, int N, int stride) {
  int base = blockIdx.z * stride;
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col > row && col < N) {
    a[base + row * N + col] = -INFINITY;
  }
}

namespace {
float* score = nullptr;
int cur_score_size = 0;
}  // namespace

#define MAX_THREADS_PER_BLOCK 1024

extern "C" {
void attention_forward(float* Q, float* K, float* V, float* O, int batch,
                       int heads, int seq_len, int head_dim) {
  size_t required_score_size =
      batch * heads * seq_len * seq_len * sizeof(float);
  if (score == nullptr || required_score_size > cur_score_size) {
    if (score != nullptr) {
      cudaFree(score);
    }
    cur_score_size = 1;
    while (cur_score_size < required_score_size) {
      cur_score_size <<= 1;
    }
    global_exit_guard.Register([]() { cudaFree(score); });
    cudaMalloc(&score, cur_score_size);
    cur_score_size = required_score_size;
  }
  float alpha = 1.0f;
  float beta = 0.0f;
  float inv_sqrt_head_dim = sqrt(1.0f / head_dim);
  cublasGemmStridedBatchedEx(get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                             seq_len, seq_len, head_dim, &inv_sqrt_head_dim, K,
                             CUDA_R_32F, head_dim, seq_len * head_dim, Q,
                             CUDA_R_32F, head_dim, seq_len * head_dim, &beta,
                             score, CUDA_R_32F, seq_len, seq_len * seq_len,
                             batch * heads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  if (seq_len <= MAX_THREADS_PER_BLOCK) {
    apply_causal_mask<<<dim3(1, seq_len, batch * heads), seq_len>>>(
        score, seq_len, seq_len * seq_len);
  } else {
    apply_causal_mask<<<dim3((seq_len + MAX_THREADS_PER_BLOCK - 1) /
                                 MAX_THREADS_PER_BLOCK,
                             seq_len, batch * heads),
                        MAX_THREADS_PER_BLOCK>>>(score, seq_len,
                                                 seq_len * seq_len);
  }
  softmax(score, score, seq_len * batch * heads, seq_len);
  cublasGemmStridedBatchedEx(
      get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len,
      &alpha, V, CUDA_R_32F, head_dim, seq_len * head_dim, score, CUDA_R_32F,
      seq_len, seq_len * seq_len, &beta, O, CUDA_R_32F, head_dim,
      seq_len * head_dim, batch * heads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
}
