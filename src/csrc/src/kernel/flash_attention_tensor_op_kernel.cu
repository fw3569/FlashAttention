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

#include "dispatch_utils.h"
#include "flash_attention_tensor_op_kernel.cuh"
#include "ldsm.cuh"

#define WARP_PER_BLOCK 4
#define ROW_PER_WARP 16
#define BR (ROW_PER_WARP * WARP_PER_BLOCK)
#define BC 128
#define STRIDE 64
#define SM_COUNT 108
#define MIN_BLOCKS_PER_SM 2

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
    OperatorShape, Operator::ElementA, Operator::LayoutA, Operator::ElementB,
    Operator::LayoutB, Operator::ElementC, Operator::LayoutC, Policy>;

using WarpMmaSVO = cutlass::gemm::warp::MmaTensorOp<
    OperatorShape, Operator::ElementA, Operator::LayoutA, Operator::ElementB,
    cutlass::layout::RowMajor, Operator::ElementC, Operator::LayoutC, Policy>;
using FragIterSVO = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShapeSVO, OperatorShape, ElementC,
    cutlass::Array<ElementC, Operator::FragmentC::kElements>, LayoutC>;
using TileIterSVO =
    cutlass::epilogue::warp::TileIteratorTensorOp<WarpShapeSVO, OperatorShape,
                                                  ElementC, LayoutC>;

constexpr unsigned int kElementsPerAccess = 2;
constexpr unsigned int kRowsPerIteration = 8;
constexpr unsigned int kLanesInQuad = 4;
constexpr unsigned int kAccessPerIteration = (WarpShapeQKS::kN >> 3);
constexpr unsigned int kElementPerIteration =
    kElementsPerAccess * kAccessPerIteration;
constexpr unsigned int kRowsPerQuad = (ROW_PER_WARP / kRowsPerIteration);

template <typename WarpShape>
struct FragmentCoord {
  static CUTLASS_DEVICE cutlass::MatrixCoord get_element_coord(
      unsigned int element_id, unsigned int lane_id) {
    using OperatorCount = cutlass::MatrixShape<
        (WarpShape::kM + OperatorShape::kM - 1) / OperatorShape::kM,
        (WarpShape::kN + OperatorShape::kN - 1) / OperatorShape::kN>;
    int thread_row = lane_id / kLanesInQuad;
    int thread_col = (lane_id & (kLanesInQuad - 1)) * kElementsPerAccess;
    int access_id = element_id / kElementsPerAccess;
    int access_row = (access_id & (kRowsPerQuad - 1)) * kRowsPerIteration;
    int access_col = (access_id / kRowsPerQuad) << 3;
    int row = 0;
    int col = element_id & (kElementsPerAccess - 1);
    return cutlass::MatrixCoord{thread_row + access_row + row,
                                thread_col + access_col + col};
  }
};

template <int max_head_dim = 128>
struct Smem {
  alignas(128) cutlass::half_t s_br_d[BR][(max_head_dim + 8)];
  alignas(128) cutlass::half_t s_bc_st[2][BC][STRIDE + 8];
};

inline int get_split_num(int batch, int heads, int seq_len, int max_split_num) {
  if (seq_len <= (BC << 2)) {
    return 1;
  }
  constexpr float kBlockSlot = SM_COUNT * MIN_BLOCKS_PER_SM;
  int grid_row = ((seq_len + BR - 1) / BR);
  int grid_col = ((seq_len + BC - 1) / BC);
  int blockz = batch * heads * grid_row;
  int qo_len = batch * heads * seq_len;
  int requested_kvlen = blockz * seq_len;
  // assume compute cost 3, load cost 1
  int min_cost = requested_kvlen * 3 + qo_len * 2;
  auto compute_utilization = [&](int split_num) -> float {
    int split_len = (grid_col + split_num - 1) / split_num * BC;
    int waves = ceilf(blockz * split_num / kBlockSlot);
    float actual_kvlen = waves * kBlockSlot * split_len;
    float actual_cost =
        actual_kvlen * 3 + qo_len * (1 + (split_num == 1 ? 1 : 3 * split_num));
    return min_cost / actual_cost;
  };
  float max_utilization = compute_utilization(1);
  int best_split_num = 1;
  for (int split_num = 2; split_num < max_split_num; ++split_num) {
    float utilization = compute_utilization(split_num);
    if (utilization >= max_utilization) {
      max_utilization = utilization;
      best_split_num = split_num;
    }
  }
  return best_split_num;
}
}  // namespace

template <bool aligned_block, int max_head_dim, bool is_causal, bool is_split>
__global__ void __launch_bounds__(WARP_PER_BLOCK * 32, MIN_BLOCKS_PER_SM)
    flash_attention_kernel(cutlass::half_t* __restrict__ Q,
                           cutlass::half_t* __restrict__ K,
                           cutlass::half_t* __restrict__ V,
                           cutlass::half_t* __restrict__ O,
                           float* __restrict__ partial_m,
                           float* __restrict__ partial_l, int seq_len,
                           int head_dim, int split_len) {
  Q += blockIdx.z * seq_len * head_dim;
  K += blockIdx.z * seq_len * head_dim;
  V += blockIdx.z * seq_len * head_dim;
  if constexpr (is_split) {
    O += blockIdx.z * seq_len * head_dim * gridDim.x +
         blockIdx.x * seq_len * head_dim;
    partial_m += blockIdx.z * seq_len * gridDim.x + blockIdx.x * seq_len;
    partial_l += blockIdx.z * seq_len * gridDim.x + blockIdx.x * seq_len;
  } else {
    O += blockIdx.z * seq_len * head_dim;
  }
  constexpr int kgroups = max_head_dim / STRIDE;
  extern __shared__ __align__(128) float raw_smem[];
  Smem<max_head_dim>& smem = reinterpret_cast<Smem<max_head_dim>(&)>(raw_smem);
  WarpMmaQKS mma_qks;
  WarpMmaSVO mma_svo;
  cutlass::Array<float, ROW_PER_WARP * STRIDE / 32> frag_o[kgroups];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kgroups; ++i) {
    frag_o[i].clear();
  }
  float reg_m[kRowsPerQuad];
  float reg_expdiffm[kRowsPerQuad];
  float reg_l[kRowsPerQuad];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kRowsPerQuad; ++i) {
    reg_m[i] = -FLT_MAX;
    reg_l[i] = 0;
  }
  ElementAccum inv_sqrt_head_dim =
      aligned_block ? (ElementAccum)rsqrtf((float)max_head_dim)
                    : (ElementAccum)rsqrtf((float)head_dim);
  int tilerow = blockIdx.y * BR;
  int warprow = threadIdx.y * ROW_PER_WARP;

  // load q
  CUTLASS_PRAGMA_UNROLL
  for (unsigned int offset = 0; offset < ROW_PER_WARP * max_head_dim;
       offset += 32 * 8) {
    int col = (offset + threadIdx.x * 8) & (max_head_dim - 1);
    int row = (offset + threadIdx.x * 8) / max_head_dim + warprow;
    if (aligned_block || tilerow + row < seq_len && col < head_dim) {
      __pipeline_memcpy_async((void*)(smem.s_br_d[row] + col),
                              (void*)(Q + (tilerow + row) * head_dim + col),
                              16);
    } else {
      __pipeline_memcpy_async((void*)(smem.s_br_d[row] + col), nullptr, 16, 16);
    }
  }

  int cache_id = 0;
  int start_tc = (is_split ? blockIdx.x * split_len : 0);
  int end_tc =
      (is_split ? min((blockIdx.x + 1) * split_len, seq_len) : seq_len);
  {
    // load k
#if (WARP_PER_BLOCK * 32 * 8 < BC * STRIDE)
    CUTLASS_PRAGMA_UNROLL
    for (unsigned int offset = 0; offset < BC * STRIDE;
         offset += WARP_PER_BLOCK * 32 * 8) {
      int col =
          ((offset + (threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
      int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 8) != 0)
      if (row >= BC) {
        break;
      }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 8 > BC * STRIDE)
    if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 8)) {
#else
    {
#endif
      int col = (((threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
      int row = ((threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#endif
      if (aligned_block || start_tc + row < seq_len && col < head_dim) {
        __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                (void*)(K + (start_tc + row) * head_dim + col),
                                16);
      } else {
        __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                nullptr, 16, 16);
      }
    }
    __pipeline_commit();
    cache_id ^= 1;
  }
  for (int tilecol = start_tc; tilecol < end_tc; tilecol += BC) {
    if (is_causal && tilecol >= tilerow + BR) {
      break;
    }
    cutlass::Array<float, ROW_PER_WARP * BC / 32> frag_s;
    frag_s.clear();
    // mul k
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < max_head_dim; k += STRIDE) {
      if (k + STRIDE < max_head_dim) {
#if (WARP_PER_BLOCK * 32 * 8 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 8) {
          int col =
              ((offset + (threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 8) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 8 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 8)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#endif
          if (aligned_block ||
              tilecol + row < seq_len && k + STRIDE + col < head_dim) {
            __pipeline_memcpy_async(
                (void*)(smem.s_bc_st[cache_id][row] + col),
                (void*)(K + (tilecol + row) * head_dim + k + STRIDE + col), 16);
          } else {
            __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                    nullptr, 16, 16);
          }
        }
      } else {
        // load v
#if (WARP_PER_BLOCK * 32 * 8 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 8) {
          int col =
              ((offset + (threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 8) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 8 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 8)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#endif
          if (aligned_block || tilecol + row < seq_len && col < head_dim) {
            __pipeline_memcpy_async(
                (void*)(smem.s_bc_st[cache_id][row] + col),
                (void*)(V + (tilecol + row) * head_dim + col), 16);
          } else {
            __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                    nullptr, 16, 16);
          }
        }
      }
      __pipeline_commit();
      cache_id ^= 1;
      __pipeline_wait_prior(1);
      __syncthreads();
      CUTLASS_PRAGMA_UNROLL
      for (int kk = 0; kk < STRIDE; kk += 16) {
        WarpMmaQKS::FragmentA frag_q[ROW_PER_WARP / 16]
                                    [16 / WarpMmaQKS::Shape::kK]
                                    [16 / WarpMmaQKS::Shape::kM];
        WarpMmaQKS::FragmentB frag_k[BC / 16][16 / WarpMmaQKS::Shape::kK]
                                    [16 / WarpMmaQKS::Shape::kN];
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t * 16 < ROW_PER_WARP; ++t) {
          __ldsm<cutlass::layout::RowMajor, 4>(
              *(reinterpret_cast<cutlass::Array<unsigned, 4>*>(
                  &frag_q[t][0][0])),
              smem.s_br_d[warprow + t * 16] + k + kk, max_head_dim + 8,
              threadIdx.x);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t * 16 < BC; ++t) {
          __ldsm<cutlass::layout::RowMajor, 4>(
              *(reinterpret_cast<cutlass::Array<unsigned, 4>*>(
                  &frag_k[t][0][0])),
              smem.s_bc_st[cache_id][t * 16] + kk, STRIDE + 8, threadIdx.x);
        };
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m * 16 < ROW_PER_WARP; ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n * 16 < BC; ++n) {
            CUTLASS_PRAGMA_UNROLL
            for (int o = 0; o * WarpMmaQKS::Shape::kK < 16; ++o) {
              mma_qks(*reinterpret_cast<WarpMmaQKS::FragmentC*>(
                          &frag_s[((n * 2) * (ROW_PER_WARP / 16) + m) * 4]),
                      frag_q[m][o][0], frag_k[n][o][0],
                      *reinterpret_cast<WarpMmaQKS::FragmentC*>(
                          &frag_s[((n * 2) * (ROW_PER_WARP / 16) + m) * 4]));
              mma_qks(
                  *reinterpret_cast<WarpMmaQKS::FragmentC*>(
                      &frag_s[((n * 2 + 1) * (ROW_PER_WARP / 16) + m) * 4]),
                  frag_q[m][o][0], frag_k[n][o][1],
                  *reinterpret_cast<WarpMmaQKS::FragmentC*>(
                      &frag_s[((n * 2 + 1) * (ROW_PER_WARP / 16) + m) * 4]));
            }
          }
        }
      }
      __syncthreads();
    }
    frag_s = frag_s * inv_sqrt_head_dim;

    // causal mask
    if constexpr (is_causal) {
      if (tilerow <= tilecol && tilecol < tilerow + BR ||
          tilecol <= tilerow && tilerow < tilecol + BC) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < frag_s.kElements; ++i) {
          auto coord =
              FragmentCoord<WarpShapeQKS>::get_element_coord(i, threadIdx.x);
          if (tilecol + coord.column() > tilerow + warprow + coord.row()) {
            frag_s[i] = -INFINITY;
          }
        }
      }
    }

    /* if (!is_causal || tilecol < tilerow + warprow + WARP_PER_BLOCK) */ {
      // softmax
      int thread_row = threadIdx.x / kLanesInQuad + warprow + tilerow;
      int thread_col =
          (threadIdx.x & (kLanesInQuad - 1)) * kElementsPerAccess + tilecol;
      int frag_id[kElementPerIteration];
      CUTLASS_PRAGMA_UNROLL
      for (int idx_in_row = 0; idx_in_row < kElementPerIteration;
           ++idx_in_row) {
        frag_id[idx_in_row] =
            (idx_in_row & ~(kElementsPerAccess - 1)) * kRowsPerQuad +
            (idx_in_row & (kElementsPerAccess - 1));
      }
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < kRowsPerQuad; ++m) {
        int row = m * kRowsPerIteration + thread_row;
        float new_m = reg_m[m];
        CUTLASS_PRAGMA_UNROLL
        for (int idx_in_row = 0; idx_in_row < kElementPerIteration;
             ++idx_in_row) {
          new_m = fmaxf(new_m,
                        frag_s[m * kElementsPerAccess + frag_id[idx_in_row]]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = kLanesInQuad >> 1; i >= 1; i >>= 1) {
          new_m = fmaxf(new_m, __shfl_xor_sync(0xffffffff, new_m, i));
        }
        float new_l = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int idx_in_row = 0; idx_in_row < kElementPerIteration;
             ++idx_in_row) {
          int id_in_frag = m * kElementsPerAccess + frag_id[idx_in_row];
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

      // find row of fragment c and rescale exp
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < kgroups; ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0, m = 0; i < frag_o[0].kElements;
             i += kElementsPerAccess, ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kElementsPerAccess; ++j) {
            frag_o[k][i + j] *= reg_expdiffm[m % kRowsPerQuad];
          }
        }
      }
    }

    // mul v
    cutlass::Array<ElementC, frag_s.kElements> frag_s16;
    cutlass::NumericArrayConverter<ElementC, ElementAccum, frag_s.kElements>
        convert;
    frag_s16 = convert(frag_s);
    WarpMmaSVO::FragmentA* frag_s16_ptr =
        reinterpret_cast<WarpMmaSVO::FragmentA*>(&frag_s16);
    CUTLASS_PRAGMA_UNROLL
    for (unsigned int k = 0; k < kgroups; ++k) {
      // transpose, tensor mma need ColumnMajor B
      if (k + 1 < kgroups) {
#if (WARP_PER_BLOCK * 32 * 8 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 8) {
          int col =
              ((offset + (threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 8) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 8 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 8)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#endif
          if (aligned_block ||
              tilecol + row < seq_len && (k + 1) * STRIDE + col < head_dim) {
            __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                    (void*)(V + (tilecol + row) * head_dim +
                                            (k + 1) * STRIDE + col),
                                    16);
          } else {
            __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                    nullptr, 16, 16);
          }
        }
      } else if (tilecol + BC < end_tc) {
        // load k
#if (WARP_PER_BLOCK * 32 * 8 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 8) {
          int col =
              ((offset + (threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 8) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 8 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 8)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 8) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 8) / STRIDE;
#endif
          if (aligned_block || tilecol + BC + row < seq_len && col < head_dim) {
            __pipeline_memcpy_async(
                (void*)(smem.s_bc_st[cache_id][row] + col),
                (void*)(K + (tilecol + BC + row) * head_dim + col), 16);
          } else {
            __pipeline_memcpy_async((void*)(smem.s_bc_st[cache_id][row] + col),
                                    nullptr, 16, 16);
          }
        }
      }
      __pipeline_commit();
      cache_id ^= 1;
      __pipeline_wait_prior(1);
      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int kk = 0; kk < BC / 16; ++kk) {
        WarpMmaSVO::FragmentB frag_v[STRIDE / 16][16 / WarpMmaSVO::Shape::kN]
                                    [16 / WarpMmaSVO::Shape::kK];
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t * 16 < STRIDE; ++t) {
          __ldsm<cutlass::layout::ColumnMajor, 4>(
              *(reinterpret_cast<cutlass::Array<unsigned, 4>*>(
                  &frag_v[t][0][0])),
              smem.s_bc_st[cache_id][kk * 16] + t * 16, STRIDE + 8,
              threadIdx.x);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m * 16 < ROW_PER_WARP; ++m) {
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n * 16 < STRIDE; ++n) {
            CUTLASS_PRAGMA_UNROLL
            for (int o = 0; o * WarpMmaSVO::Shape::kK < 16; ++o) {
              mma_svo(*reinterpret_cast<WarpMmaSVO::FragmentC*>(
                          &frag_o[k][((n * 2) * (ROW_PER_WARP / 16) + m) * 4]),
                      frag_s16_ptr[(kk * 2 + o) * (ROW_PER_WARP / 16) + m],
                      frag_v[n][0][o],
                      *reinterpret_cast<WarpMmaSVO::FragmentC*>(
                          &frag_o[k][((n * 2) * (ROW_PER_WARP / 16) + m) * 4]));
              mma_svo(
                  *reinterpret_cast<WarpMmaSVO::FragmentC*>(
                      &frag_o[k][((n * 2 + 1) * (ROW_PER_WARP / 16) + m) * 4]),
                  frag_s16_ptr[(kk * 2 + o) * (ROW_PER_WARP / 16) + m],
                  frag_v[n][1][o],
                  *reinterpret_cast<WarpMmaSVO::FragmentC*>(
                      &frag_o[k][((n * 2 + 1) * (ROW_PER_WARP / 16) + m) * 4]));
            }
          }
        }
      }
      __syncthreads();
    }
  }

  // find row of fragment c and rescale l
  if constexpr (!is_split) {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kRowsPerQuad; ++m) {
      reg_l[m] = 1.f / reg_l[m];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kgroups; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0, m = 0; i < frag_o[0].kElements;
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
    cutlass::NumericArrayConverter<ElementC, ElementAccum, frag_o[0].kElements>
        convert;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kgroups; ++k) {
      cutlass::Array<ElementC, frag_o[0].kElements> frag_o16;
      frag_o16 = convert(frag_o[k]);
      FragIterSVO frag_iter(frag_o16);
      TileIterSVO::TensorRef ref_o(smem.s_br_d[warprow] + k * STRIDE,
                                   LayoutC((max_head_dim + 8)));
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
    CUTLASS_PRAGMA_UNROLL
    for (unsigned int offset = 0; offset < ROW_PER_WARP * max_head_dim;
         offset += 32 * 8) {
      int col = ((offset + threadIdx.x * 8) & (max_head_dim - 1));
      int row = (offset + threadIdx.x * 8) / max_head_dim + warprow;
      if (aligned_block || tilerow + row < seq_len && col < head_dim) {
        *(float4*)(O + (tilerow + row) * head_dim + col) =
            *(float4*)(smem.s_br_d[row] + col);
      }
    }
  }

  // store partial_m, partial_l, [split_num, seqlen]
  if constexpr (is_split) {
    if (threadIdx.x % kLanesInQuad < ROW_PER_WARP / kRowsPerIteration) {
      partial_m[tilerow + warprow +
                threadIdx.x % kLanesInQuad * kRowsPerIteration +
                threadIdx.x / kLanesInQuad] = reg_m[threadIdx.x % kLanesInQuad];
      partial_l[tilerow + warprow +
                threadIdx.x % kLanesInQuad * kRowsPerIteration +
                threadIdx.x / kLanesInQuad] = reg_l[threadIdx.x % kLanesInQuad];
    }
  }
}

template <bool aligned_block, int max_split_num, int max_head_dim>
__global__ void __launch_bounds__(32)
    flash_attention_combine(cutlass::half_t* __restrict__ partial_o,
                            float* __restrict__ partial_m,
                            float* __restrict__ partial_l,
                            cutlass::half_t* __restrict__ O, int seq_len,
                            int head_dim, int split_num) {
  O += blockIdx.z * seq_len * head_dim;
  partial_o += blockIdx.z * seq_len * head_dim * split_num;
  partial_m += blockIdx.z * seq_len * split_num;
  partial_l += blockIdx.z * seq_len * split_num;
  constexpr int br = 32;
  constexpr int tr = (32 << 3) / max_head_dim;
  int tilerow = br * blockIdx.y;
  __shared__ __align__(128) float s_snum_br[max_split_num][br];

  if (aligned_block || tilerow + threadIdx.x < seq_len) {
    float reg_m[max_split_num];
    float scale[max_split_num];
    float reg_m_acc = -INFINITY;
    float reg_l_acc = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      if (split_id < split_num) {
        reg_m[split_id] = partial_m[split_id * seq_len + tilerow + threadIdx.x];
      } else {
        reg_m[split_id] = -INFINITY;
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      reg_m_acc = fmaxf(reg_m_acc, reg_m[split_id]);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      scale[split_id] = expf(reg_m[split_id] - reg_m_acc);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      if (split_id < split_num) {
        reg_l_acc += partial_l[split_id * seq_len + tilerow + threadIdx.x] *
                     scale[split_id];
      }
    }
    reg_l_acc = 1.0 / reg_l_acc;
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      s_snum_br[split_id][threadIdx.x] = scale[split_id] * reg_l_acc;
    }
  }
  __syncthreads();

  CUTLASS_PRAGMA_UNROLL
  for (int wr = 0; wr < br; wr += tr) {
    int row = wr + (threadIdx.x << 3) / max_head_dim;
    int col = (threadIdx.x << 3) % max_head_dim;
    if (aligned_block && row >= seq_len && col >= head_dim) {
      break;
    }
    float o_acc[8] = {};
    cutlass::half_t reg_o[8];
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      if (split_id >= split_num) {
        break;
      }
      *(float4*)(&reg_o[0]) = *(
          float4*)(&partial_o[split_id * (seq_len * head_dim) +
                              (tilerow + wr) * head_dim + (threadIdx.x << 3)]);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        o_acc[i] += reg_o[i] * s_snum_br[split_id][row];
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      reg_o[i] = cutlass::half_t(o_acc[i]);
    }
    *(float4*)(&O[(tilerow + wr) * head_dim + (threadIdx.x << 3)]) =
        *(float4*)(&reg_o[0]);
  }
}

template <bool aligned_block, int max_head_dim, bool is_causal, bool is_split>
inline void flash_attention_dispatcher(const torch::Tensor& Q,
                                       const torch::Tensor& K,
                                       const torch::Tensor& V, torch::Tensor& O,
                                       int batch, int heads, int seq_len,
                                       int head_dim, int split_num) {
  dim3 block(32, WARP_PER_BLOCK);
  dim3 grid(split_num, (seq_len + BR - 1) / BR, batch * heads);
  int smem_size = sizeof(Smem<max_head_dim>);
  cudaFuncSetAttribute(
      flash_attention_kernel<aligned_block, max_head_dim, is_causal, is_split>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if constexpr (!is_split) {
    flash_attention_kernel<aligned_block, max_head_dim, is_causal, is_split>
        <<<grid, block, smem_size>>>(
            static_cast<cutlass::half_t*>(Q.data_ptr()),
            static_cast<cutlass::half_t*>(K.data_ptr()),
            static_cast<cutlass::half_t*>(V.data_ptr()),
            static_cast<cutlass::half_t*>(O.data_ptr()), nullptr, nullptr,
            seq_len, head_dim, seq_len);
  } else {
    int split_len =
        (((seq_len + BC - 1) / BC) + split_num - 1) / split_num * BC;
    auto partial_o =
        torch::empty({batch, heads, split_num, seq_len, head_dim}, Q.options());
    auto option = Q.options();
    option = option.dtype(at::ScalarType::Float);
    auto partial_m = torch::empty({batch, heads, split_num, seq_len}, option);
    auto partial_l = torch::empty({batch, heads, split_num, seq_len}, option);
    flash_attention_kernel<aligned_block, max_head_dim, is_causal, is_split>
        <<<grid, block, smem_size>>>(
            static_cast<cutlass::half_t*>(Q.data_ptr()),
            static_cast<cutlass::half_t*>(K.data_ptr()),
            static_cast<cutlass::half_t*>(V.data_ptr()),
            static_cast<cutlass::half_t*>(partial_o.data_ptr()),
            static_cast<float*>(partial_m.data_ptr()),
            static_cast<float*>(partial_l.data_ptr()), seq_len, head_dim,
            split_len);

    bool aligned_block_combine =
        (seq_len % 32 == 0 && head_dim == max_head_dim);
    INT_DISPATCHER_6(kSplitNum, split_num, 1, 2, 4, 8, 16, 32, [&]() {
      BOOL_DISPATCHER(kAlignedBlock, aligned_block_combine, [&]() {
        flash_attention_combine<kAlignedBlock, kSplitNum, max_head_dim>
            <<<dim3(1, (seq_len + 32 - 1) / 32, batch * heads), dim3(32)>>>(
                static_cast<cutlass::half_t*>(partial_o.data_ptr()),
                static_cast<float*>(partial_m.data_ptr()),
                static_cast<float*>(partial_l.data_ptr()),
                static_cast<cutlass::half_t*>(O.data_ptr()), seq_len, head_dim,
                split_num);
      })
    });
  }
}

torch::Tensor flash_attention_tensor_op_forward(const torch::Tensor& Q,
                                                const torch::Tensor& K,
                                                const torch::Tensor& V) {
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
  bool aligned = (head_dim % 8 == 0 && seq_len % 8 == 0 &&
                  reinterpret_cast<long long>(Q.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<long long>(K.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<long long>(V.data_ptr()) % 16 == 0);
  // flaot4 load
  TORCH_CHECK(aligned, "must align to 16bit");
  // reduce kernel load o
  TORCH_CHECK(head_dim <= 256, "unsupport head dim > 256");

  auto O = torch::empty_like(Q);
  bool aligned_block =
      (seq_len % BR == 0 && seq_len % BC == 0 && head_dim % 64 == 0);
  bool is_causal = true;
  int split_num = get_split_num(batch, heads, seq_len, 32);

  INT_DISPATCHER_2(kMaxHeadDim, head_dim, 64, 128, [&]() {
    BOOL_DISPATCHER(kAlignedBlock, aligned_block, [&]() {
      BOOL_DISPATCHER(kIsCausal, is_causal, [&]() {
        BOOL_DISPATCHER(kIsSplit, split_num > 1, [&]() {
          flash_attention_dispatcher<kAlignedBlock, kMaxHeadDim, kIsCausal,
                                     kIsSplit>(Q, K, V, O, batch, heads,
                                               seq_len, head_dim, split_num);
        })
      })
    })
  });
  return O;
}
