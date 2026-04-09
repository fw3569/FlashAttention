#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>
// keep include order
#include <cutlass/epilogue/warp/fragment_iterator_simt.h>
#include <cutlass/epilogue/warp/tile_iterator_simt.h>
#include <cutlass/gemm/warp/mma_simt.h>
#include <nvrtc.h>

#include "flash_attention_kernel.cuh"

#define WARP_PER_BLOCK 4
#define ROW_PER_WARP 8
#define BR (ROW_PER_WARP * WARP_PER_BLOCK)
#define BC 64
#define STRIDE 32
#define KGROUPS 2
#define MAX_HEAD_DIM (STRIDE * KGROUPS)

namespace {
using ElementQ = float;
using ElementK = float;
using ElementV = float;
using ElementS = float;
using ElementO = float;
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutS = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

using WarpShapeQKS = cutlass::gemm::GemmShape<ROW_PER_WARP, BC, STRIDE>;
using WarpShapeSVO = cutlass::gemm::GemmShape<ROW_PER_WARP, STRIDE, BC>;
using WarpThreadArrangement = cutlass::MatrixShape<4, 8>;
using ThreadShape = cutlass::gemm::GemmShape<1, 1, 1>;
using Policy =
    cutlass::gemm::warp::MmaSimtPolicy<WarpThreadArrangement,
                                       cutlass::layout::RowMajor, ThreadShape>;

using WarpMmaQKS =
    cutlass::gemm::warp::MmaSimt<WarpShapeQKS, ElementQ, LayoutQ, ElementK,
                                 LayoutK, ElementS, LayoutS, Policy>;
using FragIterQKS = cutlass::epilogue::warp::FragmentIteratorSimt<
    WarpShapeQKS, WarpMmaQKS::ArchMmaOperator, LayoutS, Policy>;
using TileIterQKS = cutlass::epilogue::warp::TileIteratorSimt<
    WarpShapeQKS, WarpMmaQKS::ArchMmaOperator, ElementS, LayoutS,
    WarpMmaQKS::Policy>;

using WarpMmaSVO =
    cutlass::gemm::warp::MmaSimt<WarpShapeSVO, ElementS, LayoutS, ElementV,
                                 LayoutV, ElementO, LayoutO, Policy>;
using FragIterSVO = cutlass::epilogue::warp::FragmentIteratorSimt<
    WarpShapeSVO, WarpMmaSVO::ArchMmaOperator, LayoutO, Policy>;
using TileIterSVO = cutlass::epilogue::warp::TileIteratorSimt<
    WarpShapeSVO, WarpMmaSVO::ArchMmaOperator, ElementO, LayoutO,
    WarpMmaSVO::Policy>;

template <typename WarpShape, typename Policy>
struct MmaSimtRowIndex {
  static CUTLASS_DEVICE int row(unsigned int element_idx,
                                unsigned int lane_id) {
    constexpr unsigned int kElementsPerRow =
        WarpShape::kN / Policy::WarpShape::kColumn;
    return element_idx / kElementsPerRow * Policy::WarpShape::kRow +
           lane_id / Policy::WarpShape::kColumn;
  }
};
}  // namespace

__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O,
                                       int seq_len, int head_dim,
                                       bool is_causal = true) {
  Q += blockIdx.z * seq_len * head_dim;
  K += blockIdx.z * seq_len * head_dim;
  V += blockIdx.z * seq_len * head_dim;
  O += blockIdx.z * seq_len * head_dim;
  __shared__ alignas(128) float s_br_d[BR][(MAX_HEAD_DIM + 4)];
  __shared__ alignas(128) float s_bc_st[BC][(STRIDE + 4)];
  __shared__ alignas(128) float s_br_bc[BR][(BC + 4)];
  WarpMmaQKS mma_qks;
  WarpMmaSVO mma_svo;
  WarpMmaSVO::FragmentC frag_o[KGROUPS];
  for (int i = 0; i < KGROUPS; ++i) {
    frag_o[i].clear();
  }
  float reg_m = -INFINITY;
  float reg_new_m;
  float reg_new_l;
  float reg_expdiffm = 1;
  float reg_l = 0;
  float inv_sqrt_head_dim = rsqrtf((float)head_dim);

  // load q
  int tilerow = blockIdx.y * BR;
  for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 4;
       offset < BR * MAX_HEAD_DIM; offset += WARP_PER_BLOCK * 32 * 4) {
    int col = offset & (MAX_HEAD_DIM - 1);
    int row = offset / MAX_HEAD_DIM;
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
    WarpMmaQKS::IteratorA::TensorRef ref_q(
        s_br_d[threadIdx.y * ROW_PER_WARP],
        cutlass::layout::RowMajor((MAX_HEAD_DIM + 4)));
    WarpMmaQKS::IteratorA iter_q(ref_q, threadIdx.x);
    for (int k = 0; k < head_dim; k += STRIDE) {
      for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 4;
           offset < BC * STRIDE; offset += WARP_PER_BLOCK * 32 * 4) {
        int col = offset & (STRIDE - 1);
        int row = offset / STRIDE;
        // aligned to float4
        if (tilecol + row < seq_len && k + col < head_dim) {
          __pipeline_memcpy_async(
              (void*)(s_bc_st[row] + col),
              (void*)(K + (tilecol + row) * head_dim + k + col), 16);
        } else {
          __pipeline_memcpy_async((void*)(s_bc_st[row] + col), nullptr, 16, 16);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
      WarpMmaQKS::IteratorB::TensorRef ref_k(
          s_bc_st[0], cutlass::layout::ColumnMajor((STRIDE + 4)));
      WarpMmaQKS::IteratorB iter_k(ref_k, threadIdx.x);
      for (int kk = 0; kk < STRIDE; kk += ThreadShape::kK) {
        WarpMmaQKS::FragmentA frag_q;
        WarpMmaQKS::FragmentB frag_k;
        iter_q.load(frag_q);
        iter_k.load(frag_k);
        ++iter_q;
        ++iter_k;
        mma_qks(frag_s, frag_q, frag_k, frag_s);
      }
      __syncthreads();
    }
    frag_s = frag_s * inv_sqrt_head_dim;
    // don't store it to smem ?
    {
      FragIterQKS frag_iter(frag_s);
      TileIterQKS::TensorRef ref_s(s_br_bc[threadIdx.y * ROW_PER_WARP],
                                   cutlass::layout::RowMajor((BC + 4)));
      TileIterQKS tile_iter(ref_s, threadIdx.x);
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < FragIterQKS::kIterations; ++iter) {
        FragIterQKS::Fragment frag;
        frag_iter.load(frag, 0);
        tile_iter.store(frag);
        ++frag_iter;
        tile_iter.add_tile_offset({1, 0});
      }
    }
    if (is_causal) {
      for (int i = threadIdx.x; i < ROW_PER_WARP * BC; i += 32) {
        int col = i & (BC - 1);
        int row = i / BC + threadIdx.y * ROW_PER_WARP;
        if (tilerow + row < tilecol + col) {
          s_br_bc[row][col] = 0;
        }
      }
    }

    // softmax
    for (int row_in_smem = threadIdx.y * ROW_PER_WARP, row = 0;
         row < ROW_PER_WARP && tilerow + row_in_smem < seq_len;
         ++row, ++row_in_smem) {
      float new_m = __shfl_sync(0xffffffff, reg_m, row);
      for (int col = threadIdx.x;
           col < BC && tilecol + col < seq_len &&
           (!is_causal || tilerow + row_in_smem >= tilecol + col);
           col += 32) {
        new_m = max(new_m, s_br_bc[row_in_smem][col]);
      }
      for (int i = 16; i >= 1; i >>= 1) {
        new_m = max(new_m, __shfl_xor_sync(0xffffffff, new_m, i));
      }
      float new_l = 0.f;
      for (int col = threadIdx.x;
           col < BC && tilecol + col < seq_len &&
           (!is_causal || tilerow + row_in_smem >= tilecol + col);
           col += 32) {
        float temp_s = expf(s_br_bc[row_in_smem][col] - new_m);
        s_br_bc[row_in_smem][col] = temp_s;
        new_l += temp_s;
      }
      for (int i = 16; i >= 1; i >>= 1) {
        new_l += __shfl_xor_sync(0xffffffff, new_l, i);
      }
      if (row == threadIdx.x) {
        reg_new_m = new_m;
        reg_new_l = new_l;
      }
    }
    reg_expdiffm = expf(reg_m - reg_new_m);
    reg_m = reg_new_m;
    reg_l = reg_l * reg_expdiffm + reg_new_l;

    // find row of fragment c and rescale
    for (int k = 0; k < KGROUPS; ++k) {
      for (int i = 0; i < WarpMmaSVO::FragmentC::kElements; ++i) {
        frag_o[k][i] *= __shfl_sync(
            0xffffffff, reg_expdiffm,
            MmaSimtRowIndex<WarpShapeSVO, Policy>::row(i, threadIdx.x));
      }
    }

    // mul v
    for (unsigned int k = 0; k < KGROUPS; ++k) {
      for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 4;
           offset < BC * STRIDE; offset += WARP_PER_BLOCK * 32 * 4) {
        int col = (offset & (STRIDE - 1));
        int row = offset / STRIDE;
        if (tilecol + row < seq_len && k * STRIDE + col < head_dim) {
          __pipeline_memcpy_async(
              (void*)(s_bc_st[row] + col),
              (void*)(V + (tilecol + row) * head_dim + k * STRIDE + col), 16);
        } else {
          __pipeline_memcpy_async((void*)(s_bc_st[row] + col), nullptr, 16, 16);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
      WarpMmaSVO::IteratorA::TensorRef ref_s(
          s_br_bc[threadIdx.y * ROW_PER_WARP],
          cutlass::layout::RowMajor((BC + 4)));
      WarpMmaSVO::IteratorA iter_s(ref_s, threadIdx.x);
      WarpMmaSVO::IteratorB::TensorRef ref_v(
          s_bc_st[0], cutlass::layout::RowMajor((STRIDE + 4)));
      WarpMmaSVO::IteratorB iter_v(ref_v, threadIdx.x);
      for (int kk = 0; kk < BC; kk += ThreadShape::kK) {
        WarpMmaSVO::FragmentA frag_s;
        WarpMmaSVO::FragmentB frag_v;
        iter_s.load(frag_s);
        iter_v.load(frag_v);
        ++iter_s;
        ++iter_v;
        mma_svo(frag_o[k], frag_s, frag_v, frag_o[k]);
      }
      __syncthreads();
    }
  }

  // find row of fragment c and rescale
  reg_l = 1.f / reg_l;
  for (int k = 0; k < KGROUPS; ++k) {
    for (int i = 0; i < WarpMmaSVO::FragmentC::kElements; ++i) {
      frag_o[k][i] *= __shfl_sync(
          0xffffffff, reg_l,
          MmaSimtRowIndex<WarpShapeSVO, Policy>::row(i, threadIdx.x));
    }
  }

  // store o
  {
    int row = threadIdx.y * ROW_PER_WARP;
    for (int k = 0, col = 0; k < KGROUPS; ++k, col += STRIDE) {
      FragIterSVO frag_iter(frag_o[k]);
      TileIterSVO::TensorRef ref_o(
          s_br_d[row] + col, cutlass::layout::RowMajor((MAX_HEAD_DIM + 4)));
      TileIterSVO tile_iter(ref_o, threadIdx.x);
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < FragIterSVO::kIterations; ++iter) {
        FragIterSVO::Fragment frag;
        frag_iter.load(frag, 0);
        tile_iter.store(frag);
        ++frag_iter;
        tile_iter.add_tile_offset({1, 0});
      }
    }
    __syncthreads();
    for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 4;
         offset < BR * MAX_HEAD_DIM; offset += WARP_PER_BLOCK * 32 * 4) {
      int col = offset & (MAX_HEAD_DIM - 1);
      int row = offset / MAX_HEAD_DIM;
      if (tilerow + row < seq_len && col < head_dim) {
        __pipeline_memcpy_async((void*)(O + (tilerow + row) * head_dim + col),
                                (void*)(s_br_d[row] + col), 16);
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }
}

extern "C" {
void flash_attention_forward(float* Q, float* K, float* V, float* O, int batch,
                             int heads, int seq_len, int head_dim) {
  dim3 block(32, WARP_PER_BLOCK);
  dim3 grid(1, (seq_len + BR - 1) / BR, batch * heads);
  flash_attention_kernel<<<grid, block>>>(Q, K, V, O, seq_len, head_dim);
}
}
