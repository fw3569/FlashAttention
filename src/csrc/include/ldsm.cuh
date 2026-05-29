#pragma once
#include <cutlass/array.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

#include <cute/arch/util.hpp>

template <typename Layout, int MatrixCount>
__device__ void __ldsm(cutlass::Array<unsigned, MatrixCount>& D,
                       cutlass::half_t const* ptr, int stride, int lane_id);
template <>
__device__ void __ldsm<cutlass::layout::RowMajor, 1>(
    cutlass::Array<unsigned, 1>& D, cutlass::half_t const* ptr, int stride,
    int lane_id) {
  unsigned addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(ptr + stride * (lane_id % 8)));
  int x;
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"
               : "=r"(x)
               : "r"(addr));
  reinterpret_cast<int&>(D) = x;
}
template <>
__device__ void __ldsm<cutlass::layout::RowMajor, 2>(
    cutlass::Array<unsigned, 2>& D, cutlass::half_t const* ptr, int stride,
    int lane_id) {
  unsigned addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(ptr + stride * (lane_id % 16)));
  int x, y;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
               : "=r"(x), "=r"(y)
               : "r"(addr));
  reinterpret_cast<int2&>(D) = make_int2(x, y);
}

template <>
__device__ void __ldsm<cutlass::layout::RowMajor, 4>(
    cutlass::Array<unsigned, 4>& D, cutlass::half_t const* ptr, int stride,
    int lane_id) {
  unsigned addr = static_cast<uint32_t>(__cvta_generic_to_shared(
      ptr + stride * (lane_id % 16) + lane_id / 16 * 8));
  int x, y, z, w;
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
      : "r"(addr));
  reinterpret_cast<int4&>(D) = make_int4(x, y, z, w);
}
template <>
__device__ void __ldsm<cutlass::layout::ColumnMajor, 1>(
    cutlass::Array<unsigned, 1>& D, cutlass::half_t const* ptr, int stride,
    int lane_id) {
  unsigned addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(ptr + stride * (lane_id % 8)));
  int x;
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];"
               : "=r"(x)
               : "r"(addr));
  reinterpret_cast<int&>(D) = x;
}
template <>
__device__ void __ldsm<cutlass::layout::ColumnMajor, 2>(
    cutlass::Array<unsigned, 2>& D, cutlass::half_t const* ptr, int stride,
    int lane_id) {
  unsigned addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(ptr + stride * (lane_id % 16)));
  int x, y;
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"
               : "=r"(x), "=r"(y)
               : "r"(addr));
  reinterpret_cast<int2&>(D) = make_int2(x, y);
}

template <>
__device__ void __ldsm<cutlass::layout::ColumnMajor, 4>(
    cutlass::Array<unsigned, 4>& D, cutlass::half_t const* ptr, int stride,
    int lane_id) {
  unsigned addr = static_cast<uint32_t>(__cvta_generic_to_shared(
      ptr + stride * (lane_id % 16) + lane_id / 16 * 8));
  int x, y, z, w;
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
      : "r"(addr));
  reinterpret_cast<int4&>(D) = make_int4(x, y, z, w);
}