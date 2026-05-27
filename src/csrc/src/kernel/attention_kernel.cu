#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "attention_kernel.cuh"
#include "cublas_handle.cuh"
#include "softmax_kernel.cuh"
#include "utils.cuh"

__global__ void apply_causal_mask(float* a, int N, int stride) {
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col > row && col < N) {
    a[blockIdx.z * stride + row * N + col] = -INFINITY;
  }
}
__global__ void to_half(const float* a, __half* b, int M, int N) {
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    b[blockIdx.z * M * N + row * N + col] =
        __float2half(a[blockIdx.z * M * N + row * N + col]);
  }
}

namespace {
float* score = nullptr;
int cur_score_size = 0;
}  // namespace

#define BLOCK_SIZE 128

torch::Tensor attention_forward_16(const torch::Tensor& Q,
                                   const torch::Tensor& K,
                                   const torch::Tensor& V) {
  int batch = Q.size(0);
  int heads = Q.size(1);
  int seq_len = Q.size(2);
  int head_dim = Q.size(3);
  auto O = torch::empty({batch, heads, seq_len, head_dim}, Q.options());

  size_t required_score_size =
      batch * heads * seq_len * seq_len * (sizeof(float) + sizeof(__half));
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
  cublasGemmStridedBatchedEx(
      get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim,
      &inv_sqrt_head_dim, K.data_ptr(), CUDA_R_16F, head_dim,
      seq_len * head_dim, Q.data_ptr(), CUDA_R_16F, head_dim,
      seq_len * head_dim, &beta, score, CUDA_R_32F, seq_len, seq_len * seq_len,
      batch * heads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  apply_causal_mask<<<dim3((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, seq_len,
                           batch * heads),
                      BLOCK_SIZE>>>(score, seq_len, seq_len * seq_len);
  softmax(score, score, seq_len * batch * heads, seq_len);
  __half* socre_half =
      reinterpret_cast<__half*>(score + batch * heads * seq_len * seq_len);
  to_half<<<dim3((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, seq_len,
                 batch * heads),
            BLOCK_SIZE>>>(score, socre_half, seq_len, seq_len);
  cublasGemmStridedBatchedEx(
      get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len,
      &alpha, V.data_ptr(), CUDA_R_16F, head_dim, seq_len * head_dim,
      socre_half, CUDA_R_16F, seq_len, seq_len * seq_len, &beta, O.data_ptr(),
      CUDA_R_16F, head_dim, seq_len * head_dim, batch * heads, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
  return O;
}

torch::Tensor attention_forward_32(const torch::Tensor& Q,
                                   const torch::Tensor& K,
                                   const torch::Tensor& V) {
  int batch = Q.size(0);
  int heads = Q.size(1);
  int seq_len = Q.size(2);
  int head_dim = Q.size(3);
  auto O = torch::empty({batch, heads, seq_len, head_dim}, Q.options());

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
  cublasGemmStridedBatchedEx(
      get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim,
      &inv_sqrt_head_dim, K.data_ptr(), CUDA_R_32F, head_dim,
      seq_len * head_dim, Q.data_ptr(), CUDA_R_32F, head_dim,
      seq_len * head_dim, &beta, score, CUDA_R_32F, seq_len, seq_len * seq_len,
      batch * heads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  if (seq_len <= BLOCK_SIZE) {
    apply_causal_mask<<<dim3(1, seq_len, batch * heads), seq_len>>>(
        score, seq_len, seq_len * seq_len);
  } else {
    apply_causal_mask<<<dim3((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, seq_len,
                             batch * heads),
                        BLOCK_SIZE>>>(score, seq_len, seq_len * seq_len);
  }
  softmax(score, score, seq_len * batch * heads, seq_len);
  cublasGemmStridedBatchedEx(
      get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len,
      &alpha, V.data_ptr(), CUDA_R_32F, head_dim, seq_len * head_dim, score,
      CUDA_R_32F, seq_len, seq_len * seq_len, &beta, O.data_ptr(), CUDA_R_32F,
      head_dim, seq_len * head_dim, batch * heads, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
  return O;
}

torch::Tensor attention_forward(const torch::Tensor& Q, const torch::Tensor& K,
                                const torch::Tensor& V) {
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  if (Q.scalar_type() == torch::kFloat16) {
    return attention_forward_16(Q, K, V);
  } else if (Q.scalar_type() == torch::kFloat32) {
    return attention_forward_32(Q, K, V);
  }
  return {};
}
