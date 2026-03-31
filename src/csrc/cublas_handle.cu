#pragma once
#include <cublas_v2.h>

#include "softmax_kernel.cuh"
#include "utils.cuh"

namespace {
cublasHandle_t cublas_handle = nullptr;
}  // namespace

cublasHandle_t get_cublas_handle() {
  if (cublas_handle == nullptr) {
    global_exit_guard.Register([]() { cublasDestroy(cublas_handle); });
    cublasCreate(&cublas_handle);
  }
  return cublas_handle;
}
