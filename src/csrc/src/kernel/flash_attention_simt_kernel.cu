#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>
// keep include order
#include <cutlass/epilogue/warp/fragment_iterator_simt.h>
#include <cutlass/epilogue/warp/tile_iterator_simt.h>
#include <cutlass/gemm/warp/mma_simt.h>
#include <cutlass/matrix_coord.h>
#include <nvrtc.h>

#include "flash_attention_simt_kernel.cuh"

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
using WarpThreadArrangement = cutlass::MatrixShape<2, 16>;
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
struct SimtFragmentCoord {
  static CUTLASS_DEVICE cutlass::MatrixCoord get_element_coord(
      unsigned int element_idx, unsigned int lane_id) {
    constexpr unsigned int kElementsPerRow =
        WarpShape::kN / Policy::WarpShape::kColumn;
    return cutlass::MatrixCoord{
        int(element_idx / kElementsPerRow * Policy::WarpShape::kRow +
            lane_id / Policy::WarpShape::kColumn),
        int((element_idx & (kElementsPerRow - 1)) * Policy::WarpShape::kColumn +
            (lane_id & (Policy::WarpShape::kColumn - 1)))};
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
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < KGROUPS; ++i) {
    frag_o[i].clear();
  }
  constexpr unsigned int row_per_lane = ROW_PER_WARP / Policy::WarpShape::kRow;
  float reg_m[row_per_lane];
  float reg_expdiffm[row_per_lane];
  float reg_l[row_per_lane];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row_per_lane; ++i) {
    reg_m[i] = -INFINITY;
    reg_l[i] = 0;
  }
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
    if (is_causal) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < WarpMmaQKS::FragmentC::kElements; ++i) {
        auto coord = SimtFragmentCoord<WarpShapeQKS, Policy>::get_element_coord(
            i, threadIdx.x);
        if (tilecol + coord.column() >
            tilerow + threadIdx.y * ROW_PER_WARP + coord.row()) {
          frag_s[i] = 0;
        }
      }
    }

    // softmax
    {
      int thread_row = threadIdx.x / Policy::WarpShape::kColumn;
      int thread_col = threadIdx.x & (Policy::WarpShape::kColumn - 1);
      constexpr int row_size = WarpShapeQKS::kM / Policy::WarpShape::kRow;
      constexpr int col_size = WarpShapeQKS::kN / Policy::WarpShape::kColumn;
      for (int row = 0; row < row_size; ++row) {
        float new_m = reg_m[row];
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0;
             col < col_size &&
             (!is_causal ||
              tilerow + threadIdx.y * ROW_PER_WARP +
                      row * Policy::WarpShape::kRow + thread_row >=
                  tilecol + col * Policy::WarpShape::kColumn + thread_col);
             ++col) {
          new_m = max(new_m, frag_s[row * col_size + col]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = Policy::WarpShape::kColumn >> 1; i >= 1; i >>= 1) {
          new_m = max(new_m, __shfl_xor_sync(0xffffffff, new_m, i));
        }
        float new_l = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0;
             col < col_size &&
             (!is_causal ||
              tilerow + threadIdx.y * ROW_PER_WARP +
                      row * Policy::WarpShape::kRow + thread_row >=
                  tilecol + col * Policy::WarpShape::kColumn + thread_col);
             ++col) {
          float temp_s = expf(frag_s[row * col_size + col] - new_m);
          frag_s[row * col_size + col] = temp_s;
          new_l += temp_s;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = Policy::WarpShape::kColumn >> 1; i >= 1; i >>= 1) {
          new_l += __shfl_xor_sync(0xffffffff, new_l, i);
        }
        // WarpShape::kM/Policy::WarpShape::kRow<=Policy::WarpShape::kColumn
        reg_expdiffm[row] = expf(reg_m[row] - new_m);
        reg_m[row] = new_m;
        reg_l[row] = reg_l[row] * reg_expdiffm[row] + new_l;
      }
    }
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

    // find row of fragment c and rescale exp
    {
      constexpr unsigned int col_size = STRIDE / Policy::WarpShape::kColumn;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < KGROUPS; ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < WarpMmaSVO::FragmentC::kElements / col_size; ++i) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < col_size; ++j) {
            frag_o[k][i * col_size + j] *= reg_expdiffm[i];
          }
        }
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

  // find row of fragment c and rescale l
  {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < row_per_lane; ++i) {
      reg_l[i] = 1.f / reg_l[i];
    }
    constexpr unsigned int col_size = STRIDE / Policy::WarpShape::kColumn;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < KGROUPS; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < WarpMmaSVO::FragmentC::kElements / col_size; ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < col_size; ++j) {
          frag_o[k][i * col_size + j] *= reg_l[i];
        }
      }
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

torch::Tensor flash_attention_simt_forward(torch::Tensor Q, torch::Tensor K,
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
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      O.data_ptr<float>(), seq_len, head_dim, true);

  return O;
}
