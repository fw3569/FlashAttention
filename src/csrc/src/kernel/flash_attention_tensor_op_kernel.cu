#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/tensor_ref.h>
// keep include order
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <nvrtc.h>

#include "flash_attention_tensor_op_kernel.cuh"

#define WARP_PER_BLOCK 1
#define ROW_PER_WARP 16
#define BR (ROW_PER_WARP * WARP_PER_BLOCK)
#define BC 32
#define STRIDE 16
#define KGROUPS 4
#define MAX_HEAD_DIM (STRIDE * KGROUPS)

namespace {
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccum = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutAccum = cutlass::layout::RowMajor;

using OperatorShape = cutlass::gemm::GemmShape<16, 8, 8>;
using WarpShapeQKS =
    cutlass::gemm::GemmShape<ROW_PER_WARP, BC, OperatorShape::kK>;
using WarpShapeSVO =
    cutlass::gemm::GemmShape<ROW_PER_WARP, STRIDE, OperatorShape::kK>;
using Operator =
    cutlass::arch::Mma<OperatorShape, 32, ElementA, LayoutA, ElementB, LayoutB,
                       ElementAccum, LayoutAccum, cutlass::arch::OpMultiplyAdd>;
using Policy =
    cutlass::gemm::warp::MmaTensorOpPolicy<Operator,
                                           cutlass::MatrixShape<1, 1>>;

using WarpMmaQKS = cutlass::gemm::warp::MmaTensorOp<
    WarpShapeQKS, Operator::ElementA, Operator::LayoutA, Operator::ElementB,
    Operator::LayoutB, Operator::ElementC, Operator::LayoutC, Policy>;
using FragIterQKS = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShapeQKS, WarpMmaQKS::InstructionShape, ElementC,
    cutlass::Array<ElementC, Operator::FragmentC::kElements>, LayoutC>;
using TileIterQKS = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShapeQKS, WarpMmaQKS::InstructionShape, ElementC, LayoutC>;

using WarpMmaSVO = cutlass::gemm::warp::MmaTensorOp<
    WarpShapeSVO, Operator::ElementA, Operator::LayoutA, Operator::ElementB,
    Operator::LayoutB, Operator::ElementC, Operator::LayoutC, Policy>;
using FragIterSVO = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShapeSVO, WarpMmaSVO::InstructionShape, ElementC,
    cutlass::Array<ElementC, Operator::FragmentC::kElements>, LayoutC>;
using TileIterSVO = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShapeSVO, WarpMmaSVO::InstructionShape, ElementC, LayoutC>;

constexpr unsigned int kElementsPerAccess = 2;
constexpr unsigned int kRowsPerIteration = 8;
constexpr unsigned int kLanesInQuad = 4;
constexpr unsigned int kAccessPerIteration = (WarpShapeQKS::kN >> 3);
constexpr unsigned int kElementPerIteration =
    kElementsPerAccess * kAccessPerIteration;
constexpr unsigned int kRowsPerQuad = (ROW_PER_WARP / kRowsPerIteration);
constexpr unsigned int kMaxBcStride = std::max(BC, STRIDE);

template <typename WarpShape>
struct SimtFragmentCoord {
  static CUTLASS_DEVICE cutlass::MatrixCoord get_element_coord(
      unsigned int element_id, unsigned int lane_id) {
    using OperatorCount = cutlass::MatrixShape<
        (WarpShape::kM + OperatorShape::kM - 1) / OperatorShape::kM,
        (WarpShape::kN + OperatorShape::kN - 1) / OperatorShape::kN>;
    int thread_row = lane_id / kLanesInQuad;
    int thread_col = (lane_id & (kLanesInQuad - 1)) * kElementsPerAccess;
    int access_id = element_id / kElementsPerAccess;
    int access_row = (access_id & (kRowsPerQuad - 1)) * kRowsPerIteration;
    int access_col = access_id / kRowsPerQuad << 3;
    int row = 0;
    int col = element_id & (kElementsPerAccess - 1);
    return cutlass::MatrixCoord{thread_row + access_row + row,
                                thread_col + access_col + col};
  }
};
}  // namespace

__global__ void flash_attention_kernel(cutlass::half_t* Q, cutlass::half_t* K,
                                       cutlass::half_t* V, cutlass::half_t* O,
                                       int seq_len, int head_dim,
                                       bool is_causal = true) {
  Q += blockIdx.z * seq_len * head_dim;
  K += blockIdx.z * seq_len * head_dim;
  V += blockIdx.z * seq_len * head_dim;
  O += blockIdx.z * seq_len * head_dim;
  __shared__ alignas(128) cutlass::half_t s_br_d[BR][(MAX_HEAD_DIM + 8)];
  __shared__ alignas(128)
      cutlass::half_t s_bc_st[BC * STRIDE + kMaxBcStride * 8];
  __shared__ alignas(128) cutlass::half_t s_br_bc[BR][(BC + 8)];
  WarpMmaQKS mma_qks;
  WarpMmaSVO mma_svo;
  WarpMmaSVO::FragmentC frag_o[KGROUPS];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < KGROUPS; ++i) {
    frag_o[i].clear();
  }
  float reg_m[kRowsPerQuad];
  float reg_expdiffm[kRowsPerQuad];
  float reg_l[kRowsPerQuad];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kRowsPerQuad; ++i) {
    reg_m[i] = -INFINITY;
    reg_l[i] = 0;
  }
  ElementAccum inv_sqrt_head_dim = (ElementAccum)rsqrtf((float)head_dim);
  int tilerow = blockIdx.y * BR;
  int warprow = threadIdx.y * ROW_PER_WARP;

  // load q
  for (unsigned int offset = threadIdx.x * 8;
       offset < ROW_PER_WARP * MAX_HEAD_DIM; offset += 32 * 8) {
    int col = offset & (MAX_HEAD_DIM - 1);
    int row = offset / MAX_HEAD_DIM + warprow;
    if (tilerow + row < seq_len && col < head_dim) {
      __pipeline_memcpy_async((void*)(s_br_d[row] + col),
                              (void*)(Q + (tilerow + row) * head_dim + col),
                              16);
    } else {
      __pipeline_memcpy_async((void*)(s_br_d[row] + col), nullptr, 16, 16);
    }
  }

  // mul k
  for (int tilecol = 0;
       tilecol < seq_len && (!is_causal || tilecol < tilerow + BR);
       tilecol += BC) {
    WarpMmaQKS::FragmentC frag_s;
    frag_s.clear();
    for (int k = 0; k < head_dim; k += STRIDE) {
      for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 8;
           offset < BC * STRIDE; offset += WARP_PER_BLOCK * 32 * 8) {
        int col = offset & (STRIDE - 1);
        int row = offset / STRIDE;
        // aligned to float4
        if (tilecol + row < seq_len && k + col < head_dim) {
          __pipeline_memcpy_async(
              (void*)(s_bc_st + row * (STRIDE + 8) + col),
              (void*)(K + (tilecol + row) * head_dim + k + col), 16);
        } else {
          __pipeline_memcpy_async((void*)(s_bc_st + row * (STRIDE + 8) + col),
                                  nullptr, 16, 16);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
#pragma unroll 2
      for (int kk = 0; kk < STRIDE; kk += OperatorShape::kK) {
        WarpMmaQKS::IteratorA::TensorRef ref_q(s_br_d[warprow] + k + kk,
                                               LayoutA((MAX_HEAD_DIM + 8)));
        WarpMmaQKS::IteratorA iter_q(ref_q, threadIdx.x);
        WarpMmaQKS::IteratorB::TensorRef ref_k(s_bc_st + kk,
                                               LayoutB((STRIDE + 8)));
        WarpMmaQKS::IteratorB iter_k(ref_k, threadIdx.x);
        WarpMmaQKS::FragmentA frag_q;
        WarpMmaQKS::FragmentB frag_k;
        iter_q.load(frag_q);
        iter_k.load(frag_k);
        mma_qks(frag_s, frag_q, frag_k, frag_s);
      }
      __syncthreads();
    }
    frag_s = frag_s * inv_sqrt_head_dim;

    // causal mask
    if (is_causal) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < WarpMmaQKS::FragmentC::kElements; ++i) {
        auto coord =
            SimtFragmentCoord<WarpShapeQKS>::get_element_coord(i, threadIdx.x);
        if (tilecol + coord.column() > tilerow + warprow + coord.row()) {
          frag_s[i] = 0;
        }
      }
    }

    // softmax
    {
      int thread_row = threadIdx.x / kLanesInQuad + warprow + tilerow;
      int thread_col =
          (threadIdx.x & (kLanesInQuad - 1)) * kElementsPerAccess + tilecol;
      for (int m = 0; m < kRowsPerQuad; ++m) {
        int row = m * kRowsPerIteration + thread_row;
        float new_m = reg_m[m];
        CUTLASS_PRAGMA_UNROLL
        for (int idx_in_row = 0;
             idx_in_row < kElementPerIteration &&
             (!is_causal || row >= thread_col +
                                       (idx_in_row / kElementsPerAccess << 3) +
                                       idx_in_row % kElementsPerAccess);
             ++idx_in_row) {
          new_m = max(
              new_m,
              frag_s[m * kElementsPerAccess +
                     (idx_in_row & ~(kElementsPerAccess - 1)) * kRowsPerQuad +
                     (idx_in_row & (kElementsPerAccess - 1))]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = kLanesInQuad >> 1; i >= 1; i >>= 1) {
          new_m = max(new_m, __shfl_xor_sync(0xffffffff, new_m, i));
        }
        float new_l = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int idx_in_row = 0;
             idx_in_row < kElementPerIteration &&
             (!is_causal || row >= thread_col +
                                       (idx_in_row / kElementsPerAccess << 3) +
                                       idx_in_row % kElementsPerAccess);
             ++idx_in_row) {
          int id_in_frag =
              m * kElementsPerAccess +
              (idx_in_row & ~(kElementsPerAccess - 1)) * kRowsPerQuad +
              (idx_in_row & (kElementsPerAccess - 1));
          float temp_s = expf(frag_s[id_in_frag] - new_m);
          frag_s[id_in_frag] = temp_s;
          new_l += temp_s;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = kLanesInQuad >> 1; i >= 1; i >>= 1) {
          new_l += __shfl_xor_sync(0xffffffff, new_l, i);
        }
        reg_expdiffm[m] = expf(reg_m[m] - new_m);
        reg_m[m] = new_m;
        reg_l[m] = reg_l[m] * reg_expdiffm[m] + new_l;
      }
    }
    {
      cutlass::Array<ElementC, WarpMmaQKS::FragmentC::kElements> frag_s16;
      cutlass::NumericArrayConverter<ElementC, ElementAccum,
                                     WarpMmaQKS::FragmentC::kElements>
          convert;
      frag_s16 = convert(frag_s);

      FragIterQKS frag_iter(frag_s16);
      TileIterQKS::TensorRef ref_s(s_br_bc[warprow], LayoutC((BC + 8)));
      TileIterQKS tile_iter(ref_s, threadIdx.x);
      for (int iter = 0; iter < FragIterQKS::kIterations; ++iter) {
        FragIterQKS::Fragment frag;
        frag_iter.load(frag, 0);
        tile_iter.store(frag);
        ++frag_iter;
        tile_iter.add_tile_offset({1, 0});
      }
    }

    // find row of fragment c and rescale exp
    {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < KGROUPS; ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0, m = 0; i < WarpMmaSVO::FragmentC::kElements;
             i += kElementsPerAccess, ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kElementsPerAccess; ++j) {
            frag_o[k][i + j] *= reg_expdiffm[m % kRowsPerQuad];
          }
        }
      }
    }

    // mul v
    for (unsigned int k = 0; k < KGROUPS; ++k) {
      // transpose, tensor mma need ColumnMajor B
      for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 8;
           offset < BC * STRIDE; offset += WARP_PER_BLOCK * 32 * 8) {
        int col = (offset & (STRIDE - 1));
        int row = offset / STRIDE;
        if (tilecol + row < seq_len && k * STRIDE + col < head_dim) {
          cutlass::half_t buffer[8];
          *(float4*)(buffer) =
              *(float4*)(V + (tilecol + row) * head_dim + k * STRIDE + col);
          for (int i = 0; i < 8; ++i) {
            *(s_bc_st + (col + i) * (BC + 8) + row) = buffer[i];
          }
        } else {
          for (int i = 0; i < 8; ++i) {
            *(s_bc_st + (col + i) * (BC + 8) + row) = (cutlass::half_t)0.f;
          }
        }
      }
      __syncthreads();
#pragma unroll 2
      for (int kk = 0; kk < BC; kk += OperatorShape::kK) {
        WarpMmaSVO::IteratorA::TensorRef ref_s(s_br_bc[warprow] + kk,
                                               LayoutA((BC + 8)));
        WarpMmaSVO::IteratorA iter_s(ref_s, threadIdx.x);
        WarpMmaSVO::IteratorB::TensorRef ref_v(s_bc_st + kk, LayoutB((BC + 8)));
        WarpMmaSVO::IteratorB iter_v(ref_v, threadIdx.x);
        WarpMmaSVO::FragmentA frag_s;
        WarpMmaSVO::FragmentB frag_v;
        iter_s.load(frag_s);
        iter_v.load(frag_v);
        mma_svo(frag_o[k], frag_s, frag_v, frag_o[k]);
      }
      __syncthreads();
    }
  }

  // find row of fragment c and rescale l
  {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kRowsPerQuad; ++m) {
      reg_l[m] = 1.f / reg_l[m];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < KGROUPS; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0, m = 0; i < WarpMmaSVO::FragmentC::kElements;
           i += kElementsPerAccess, ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kElementsPerAccess; ++j) {
          frag_o[k][i + j] *= reg_l[m % kRowsPerQuad];
        }
      }
    }
  }

  // store o
  {
    cutlass::NumericArrayConverter<ElementC, ElementAccum,
                                   WarpMmaSVO::FragmentC::kElements>
        convert;
    for (int k = 0; k < KGROUPS; ++k) {
      cutlass::Array<ElementC, WarpMmaSVO::FragmentC::kElements> frag_o16;
      frag_o16 = convert(frag_o[k]);
      FragIterSVO frag_iter(frag_o16);
      TileIterSVO::TensorRef ref_o(s_br_d[warprow] + k * STRIDE,
                                   LayoutC((MAX_HEAD_DIM + 8)));
      TileIterSVO tile_iter(ref_o, threadIdx.x);
      for (int iter = 0; iter < FragIterSVO::kIterations; ++iter) {
        FragIterSVO::Fragment frag;
        frag_iter.load(frag, 0);
        tile_iter.store(frag);
        ++frag_iter;
        tile_iter.add_tile_offset({1, 0});
      }
    }
    for (unsigned int offset = threadIdx.x * 8;
         offset < ROW_PER_WARP * MAX_HEAD_DIM; offset += 32 * 8) {
      int col = offset & (MAX_HEAD_DIM - 1);
      int row = offset / MAX_HEAD_DIM + warprow;
      if (tilerow + row < seq_len && col < head_dim) {
        __pipeline_memcpy_async((void*)(O + (tilerow + row) * head_dim + col),
                                (void*)(s_br_d[row] + col), 16);
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }
}

torch::Tensor flash_attention_tensor_op_forward(torch::Tensor Q,
                                                torch::Tensor K,
                                                torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  int batch = Q.size(0);
  int heads = Q.size(1);
  int seq_len = Q.size(2);
  int head_dim = Q.size(3);

  auto O = torch::zeros({batch, heads, seq_len, head_dim}, Q.options());

  dim3 block(32, WARP_PER_BLOCK);
  dim3 grid(1, (seq_len + BR - 1) / BR, batch * heads);
  flash_attention_kernel<<<grid, block>>>(
      static_cast<cutlass::half_t*>(Q.data_ptr()),
      static_cast<cutlass::half_t*>(K.data_ptr()),
      static_cast<cutlass::half_t*>(V.data_ptr()),
      static_cast<cutlass::half_t*>(O.data_ptr()), seq_len, head_dim, true);

  return O;
}
