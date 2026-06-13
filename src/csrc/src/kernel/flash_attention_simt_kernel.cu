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
#define __max(a, b) (((a) > (b)) ? (a) : (b))
#define __min(a, b) (((a) < (b)) ? (a) : (b))

#include "dispatch_utils.h"
#include "flash_attention_simt_kernel.cuh"

#define WARP_PER_BLOCK 4
// ROW_PER_WARP <= 32, each lane store one row in softmax
#define ROW_PER_WARP 8
#define BR (ROW_PER_WARP * WARP_PER_BLOCK)
#define BC 32
#define STRIDE 32
#define SM_COUNT 108
#define MIN_BLOCKS_PER_SM 2

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

constexpr int access_size_b = __min(4, STRIDE / Policy::WarpShape::kColumn);
using access_type_b =
    std::conditional_t<access_size_b == 4, float4,
                       std::conditional_t<access_size_b == 2, float2, float>>;

template <typename WarpShape, typename Policy>
struct FragmentCoord {
  static CUTLASS_DEVICE cutlass::MatrixCoord get_element_coord(
      unsigned int element_idx, unsigned int lane_id) {
    constexpr unsigned int kElementsPerRow =
        WarpShape::kN / Policy::WarpShape::kColumn;
    int col_id = element_idx & (kElementsPerRow - 1);
    return cutlass::MatrixCoord{
        int(element_idx / kElementsPerRow * Policy::WarpShape::kRow +
            lane_id / Policy::WarpShape::kColumn),
        int(lane_id % Policy::WarpShape::kColumn * access_size_b +
            (col_id & ~(access_size_b - 1)) * Policy::WarpShape::kColumn +
            (col_id & (access_size_b - 1)))};
    // int((element_idx & (kElementsPerRow - 1)) * Policy::WarpShape::kColumn +
    //     (lane_id & (Policy::WarpShape::kColumn - 1)))};
  }
};

template <int max_head_dim = 128>
struct Smem {
  alignas(128) float s_br_d[BR][(max_head_dim + 4)];
  alignas(128) float s_bc_st[2][BC][(STRIDE + 4)];
  alignas(128) float s_br_bc[BR][BC + 4];
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
    flash_attention_kernel(float* __restrict__ Q, float* __restrict__ K,
                           float* __restrict__ V, float* __restrict__ O,
                           float* __restrict__ partial_m,
                           float* __restrict__ partial_l, int seq_len,
                           int head_dim, int split_len) {
  Q += blockIdx.z * seq_len * head_dim;
  K += blockIdx.z * seq_len * head_dim;
  V += blockIdx.z * seq_len * head_dim;
  if (is_split) {
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
  float (&s_bc_st_interval)[2][BC / access_size_b][STRIDE * access_size_b + 4] =
      *reinterpret_cast<
          float (*)[2][BC / access_size_b][STRIDE * access_size_b + 4]>(
          &smem.s_bc_st[0][0][(BC - BC / access_size_b) * 4]);
  WarpMmaQKS mma_qks;
  WarpMmaSVO mma_svo;
  cutlass::Array<float, ROW_PER_WARP * STRIDE / 32> frag_o[kgroups];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kgroups; ++i) {
    frag_o[i].clear();
  }
  constexpr unsigned int row_per_lane = ROW_PER_WARP / Policy::WarpShape::kRow;
  float reg_m[row_per_lane];
  float reg_expdiffm[row_per_lane];
  float reg_l[row_per_lane];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row_per_lane; ++i) {
    reg_m[i] = -FLT_MAX;
    reg_l[i] = 0;
  }
  float inv_sqrt_head_dim =
      aligned_block ? rsqrtf((float)max_head_dim) : rsqrtf((float)head_dim);
  int tilerow = blockIdx.y * BR;
  int warprow = threadIdx.y * ROW_PER_WARP;

  // load q
  CUTLASS_PRAGMA_UNROLL
  for (unsigned int offset = 0; offset < ROW_PER_WARP * max_head_dim;
       offset += 32 * 4) {
    int col = (offset + threadIdx.x * 4) & (max_head_dim - 1);
    int row = (offset + threadIdx.x * 4) / max_head_dim + warprow;
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
#if (WARP_PER_BLOCK * 32 * 4 < BC * STRIDE)
    CUTLASS_PRAGMA_UNROLL
    for (unsigned int offset = 0; offset < BC * STRIDE;
         offset += WARP_PER_BLOCK * 32 * 4) {
      int col = (offset + (threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1);
      int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 4) != 0)
      if (row >= BC) {
        break;
      }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 4 > BC * STRIDE)
    if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 4)) {
#else
    {
#endif
      int col = (((threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1));
      int row = ((threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#endif
      // aligned to float4
      if (aligned_block || start_tc + row < seq_len && col < head_dim) {
        __pipeline_memcpy_async(
            (void*)(s_bc_st_interval[0][row / access_size_b] +
                    (row & (access_size_b - 1)) * STRIDE + col),
            (void*)(K + (start_tc + row) * head_dim + col), 16);
      } else {
        __pipeline_memcpy_async(
            (void*)(s_bc_st_interval[0][row / access_size_b] +
                    (row & (access_size_b - 1)) * STRIDE + col),
            nullptr, 16, 16);
      }
    }
    cache_id ^= 1;
    __pipeline_commit();
  }
  for (int tilecol = start_tc; tilecol < end_tc; tilecol += BC) {
    if (is_causal && tilecol >= tilerow + BR) {
      break;
    }
    cutlass::Array<float, ROW_PER_WARP * BC / 32> frag_s;
    frag_s.clear();
    // mul k
    // WarpMmaQKS::IteratorA::TensorRef ref_q(smem.s_br_d[warprow],
    //                                        LayoutQ((max_head_dim + 4)));
    // WarpMmaQKS::IteratorA iter_q(ref_q, threadIdx.x);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < max_head_dim; k += STRIDE) {
      if (k + STRIDE < max_head_dim) {
#if (WARP_PER_BLOCK * 32 * 4 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 4) {
          int col =
              (offset + (threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1);
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 4) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 4 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 4)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#endif
          // aligned to float4
          if (aligned_block ||
              tilecol + row < seq_len && k + STRIDE + col < head_dim) {
            __pipeline_memcpy_async(
                (void*)(s_bc_st_interval[cache_id][row / access_size_b] +
                        (row & (access_size_b - 1)) * STRIDE + col),
                (void*)(K + (tilecol + row) * head_dim + k + STRIDE + col), 16);
          } else {
            __pipeline_memcpy_async(
                (void*)(s_bc_st_interval[cache_id][row / access_size_b] +
                        (row & (access_size_b - 1)) * STRIDE + col),
                nullptr, 16, 16);
          }
        }
      } else {
#if (WARP_PER_BLOCK * 32 * 4 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 4) {
          int col =
              (offset + (threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1);
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 4) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 4 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 4)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
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
      cache_id ^= 1;
      __pipeline_commit();
      __pipeline_wait_prior(1);
      __syncthreads();
      // WarpMmaQKS::IteratorB::TensorRef ref_k(smem.s_bc_st[cache_id][0],
      //                                        LayoutK((STRIDE + 4)));
      // WarpMmaQKS::IteratorB iter_k(ref_k, threadIdx.x);
      CUTLASS_PRAGMA_UNROLL
      for (int kk = 0; kk < STRIDE; kk += 4) {
        // WarpMmaQKS::FragmentA frag_q;
        // iter_q.load(frag_q);
        // ++iter_q;
        // WarpMmaQKS::FragmentB frag_k;
        // iter_k.load(frag_k);
        // ++iter_k;
        float reg_q[WarpMmaQKS::FragmentA::kElements][4];
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < WarpMmaQKS::FragmentA::kElements; ++row) {
          *(float4*)(&reg_q[row][0]) =
              *(float4*)(&smem.s_br_d[warprow +
                                      threadIdx.x / Policy::WarpShape::kColumn +
                                      row * Policy::WarpShape::kRow][k + kk]);
        }
        float reg_k[WarpMmaQKS::FragmentB::kElements][4];
        CUTLASS_PRAGMA_UNROLL
        for (int eid = 0; eid < WarpMmaQKS::FragmentB::kElements; ++eid) {
          int row = threadIdx.x % Policy::WarpShape::kColumn * access_size_b +
                    (eid & ~(access_size_b - 1)) * Policy::WarpShape::kColumn +
                    (eid & (access_size_b - 1));
          int col = kk;
          *(float4*)(&reg_k[eid][0]) =
              *(float4*)(s_bc_st_interval[cache_id][row / access_size_b] +
                         (row & (access_size_b - 1)) * STRIDE + col);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int kkk = 0; kkk < 4; ++kkk) {
          WarpMmaQKS::FragmentA frag_q;
          CUTLASS_PRAGMA_UNROLL
          for (int row = 0; row < WarpMmaQKS::FragmentA::kElements; ++row) {
            frag_q[row] = reg_q[row][kkk];
          }
          WarpMmaQKS::FragmentB frag_k;
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < WarpMmaQKS::FragmentB::kElements; ++col) {
            frag_k[col] = reg_k[col][kkk];
          }
          mma_qks(frag_s, frag_q, frag_k, frag_s);
        }
      }
      __syncthreads();
    }
    frag_s = frag_s * inv_sqrt_head_dim;
    if constexpr (is_causal) {
      if (tilerow <= tilecol && tilecol < tilerow + BR ||
          tilecol <= tilerow && tilerow < tilecol + BC) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < frag_s.kElements; ++i) {
          auto coord = FragmentCoord<WarpShapeQKS, Policy>::get_element_coord(
              i, threadIdx.x);
          if (tilecol + coord.column() > tilerow + warprow + coord.row()) {
            frag_s[i] = -INFINITY;
          }
        }
      }
    }

    // softmax
    /* if (!is_causal || tilecol < tilerow + warprow + WARP_PER_BLOCK) */ {
      int thread_row = threadIdx.x / Policy::WarpShape::kColumn;
      int thread_col = threadIdx.x & (Policy::WarpShape::kColumn - 1);
      constexpr int row_size = WarpShapeQKS::kM / Policy::WarpShape::kRow;
      constexpr int col_size = WarpShapeQKS::kN / Policy::WarpShape::kColumn;
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < row_size; ++row) {
        float new_m = reg_m[row];
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < col_size; ++col) {
          new_m = fmaxf(new_m, frag_s[row * col_size + col]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = Policy::WarpShape::kColumn >> 1; i >= 1; i >>= 1) {
          new_m = fmaxf(new_m, __shfl_xor_sync(0xffffffff, new_m, i));
        }
        float new_l = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < col_size; ++col) {
          float temp_s = expf(frag_s[row * col_size + col] - new_m);
          frag_s[row * col_size + col] = temp_s;
          new_l += temp_s;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = Policy::WarpShape::kColumn >> 1; i >= 1; i >>= 1) {
          new_l += __shfl_xor_sync(0xffffffff, new_l, i);
        }
        reg_expdiffm[row] = expf(reg_m[row] - new_m);
        reg_m[row] = new_m;
        reg_l[row] = reg_l[row] * reg_expdiffm[row] + new_l;
      }

      // find row of fragment c and rescale exp
      constexpr unsigned int col_size_o = STRIDE / Policy::WarpShape::kColumn;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < kgroups; ++k) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < frag_o[0].kElements / col_size_o; ++i) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < col_size_o; ++j) {
            frag_o[k][i * col_size_o + j] *= reg_expdiffm[i];
          }
        }
      }
    }
    {
      // FragIterQKS frag_iter(frag_s);
      // TileIterQKS::TensorRef ref_s(smem.s_br_bc[warprow], LayoutS((BC + 4)));
      // TileIterQKS tile_iter(ref_s, threadIdx.x);
      // CUTLASS_PRAGMA_UNROLL
      // for (int iter = 0; iter < FragIterQKS::kIterations; ++iter) {
      //   FragIterQKS::Fragment frag;
      //   frag_iter.load(frag, 0);
      //   tile_iter.store(frag);
      //   ++frag_iter;
      //   tile_iter.add_tile_offset({1, 0});
      // }
      constexpr unsigned int col_size = BC / Policy::WarpShape::kColumn;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < frag_s.kElements / col_size; ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < col_size; j += access_size_b) {
          int col = threadIdx.x % Policy::WarpShape::kColumn * access_size_b +
                    j * Policy::WarpShape::kColumn;
          int row = warprow + threadIdx.x / Policy::WarpShape::kColumn +
                    i * Policy::WarpShape::kRow;
          *(access_type_b*)(&smem.s_br_bc[row][col]) =
              *(access_type_b*)(&frag_s[i * col_size + j]);
        }
      }
    }

    // mul v
    CUTLASS_PRAGMA_UNROLL
    for (unsigned int k = 0; k < kgroups; ++k) {
      if (k + 1 < kgroups) {
#if (WARP_PER_BLOCK * 32 * 4 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 4) {
          int col =
              (offset + (threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1);
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 4) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 4 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 4)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
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
#if (WARP_PER_BLOCK * 32 * 4 < BC * STRIDE)
        CUTLASS_PRAGMA_UNROLL
        for (unsigned int offset = 0; offset < BC * STRIDE;
             offset += WARP_PER_BLOCK * 32 * 4) {
          int col =
              (offset + (threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1);
          int row = (offset + (threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#if ((BC * STRIDE) % (WARP_PER_BLOCK * 32 * 4) != 0)
          if (row >= BC) {
            break;
          }
#endif
#else
#if (WARP_PER_BLOCK * 32 * 4 > BC * STRIDE)
        if (threadIdx.y * 32 + threadIdx.x < (BC * STRIDE / 4)) {
#else
        {
#endif
          int col = (((threadIdx.y * 32 + threadIdx.x) * 4) & (STRIDE - 1));
          int row = ((threadIdx.y * 32 + threadIdx.x) * 4) / STRIDE;
#endif
          // aligned to float4
          if (aligned_block || tilecol + BC + row < seq_len && col < head_dim) {
            __pipeline_memcpy_async(
                (void*)(s_bc_st_interval[cache_id][row / access_size_b] +
                        (row & (access_size_b - 1)) * STRIDE + col),
                (void*)(K + (tilecol + BC + row) * head_dim + col), 16);
          } else {
            __pipeline_memcpy_async(
                (void*)(s_bc_st_interval[cache_id][row / access_size_b] +
                        (row & (access_size_b - 1)) * STRIDE + col),
                nullptr, 16, 16);
          }
        }
      }
      cache_id ^= 1;
      __pipeline_commit();
      __pipeline_wait_prior(1);
      __syncthreads();
      // WarpMmaSVO::IteratorA::TensorRef ref_s(smem.s_br_bc[warprow],
      //                                        LayoutS((BC + 4)));
      // WarpMmaSVO::IteratorA iter_s(ref_s, threadIdx.x);
      // WarpMmaSVO::IteratorB::TensorRef ref_v(smem.s_bc_st[cache_id][0],
      //                                        LayoutV((STRIDE + 4)));
      // WarpMmaSVO::IteratorB iter_v(ref_v, threadIdx.x);
      CUTLASS_PRAGMA_UNROLL
      for (int kk = 0; kk < BC; kk += 4) {
        // WarpMmaSVO::FragmentA frag_s;
        // iter_s.load(frag_s);
        // ++iter_s;
        // WarpMmaSVO::FragmentB frag_v;
        // iter_v.load(frag_v);
        // ++iter_v;
        float reg_s[WarpMmaSVO::FragmentA::kElements][4];
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < WarpMmaSVO::FragmentA::kElements; ++row) {
          *(float4*)(&reg_s[row][0]) = *(
              float4*)(&smem.s_br_bc[warprow +
                                     threadIdx.x / Policy::WarpShape::kColumn +
                                     row * Policy::WarpShape::kRow][kk]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int kkk = 0; kkk < 4; ++kkk) {
          WarpMmaSVO::FragmentA frag_s;
          CUTLASS_PRAGMA_UNROLL
          for (int row = 0; row < WarpMmaSVO::FragmentA::kElements; ++row) {
            frag_s[row] = reg_s[row][kkk];
          }
          WarpMmaSVO::FragmentB frag_v;
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < WarpMmaSVO::FragmentB::kElements;
               col += access_size_b) {
            *(access_type_b*)(&frag_v[col]) = *(
                access_type_b*)(&smem.s_bc_st[cache_id][kk + kkk]
                                             [threadIdx.x %
                                                  Policy::WarpShape::kColumn *
                                                  access_size_b +
                                              col *
                                                  Policy::WarpShape::kColumn]);
          }
          mma_svo(frag_o[k], frag_s, frag_v, frag_o[k]);
        }
      }
      __syncthreads();
    }
  }

  // find row of fragment c and rescale l
  if constexpr (!is_split) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < row_per_lane; ++i) {
      reg_l[i] = 1.f / reg_l[i];
    }
    constexpr unsigned int col_size = STRIDE / Policy::WarpShape::kColumn;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kgroups; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < frag_o[0].kElements / col_size; ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < col_size; ++j) {
          frag_o[k][i * col_size + j] *= reg_l[i];
        }
      }
    }
  }

  // store o
  {
    // CUTLASS_PRAGMA_UNROLL
    // for (int k = 0, col = 0; k < kgroups; ++k, col += STRIDE) {
    //   FragIterSVO frag_iter(frag_o[k]);
    //   TileIterSVO::TensorRef ref_o(smem.s_br_d[warprow] + col,
    //                                LayoutO((max_head_dim + 4)));
    //   TileIterSVO tile_iter(ref_o, threadIdx.x);
    //   CUTLASS_PRAGMA_UNROLL
    //   for (int iter = 0; iter < FragIterSVO::kIterations; ++iter) {
    //     FragIterSVO::Fragment frag;
    //     frag_iter.load(frag, 0);
    //     tile_iter.store(frag);
    //     ++frag_iter;
    //     tile_iter.add_tile_offset({1, 0});
    //   }
    // }
    // __syncthreads();
    // CUTLASS_PRAGMA_UNROLL
    // for (unsigned int offset = (threadIdx.y * 32 + threadIdx.x) * 4;
    //      offset < BR * max_head_dim; offset += WARP_PER_BLOCK * 32 * 4) {
    //   int col = offset & (max_head_dim - 1);
    //   int row = offset / max_head_dim;
    //   if (aligned_block || tilerow + row < seq_len && col < head_dim) {
    //     *(float4*)(O + (tilerow + row) * head_dim + col) =
    //         *(float4*)(smem.s_br_d[row] + col);
    //   }
    // }
    constexpr unsigned int col_size = STRIDE / Policy::WarpShape::kColumn;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kgroups; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < frag_o[0].kElements / col_size; ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < col_size; j += access_size_b) {
          int col = k * STRIDE +
                    threadIdx.x % Policy::WarpShape::kColumn * access_size_b +
                    j * Policy::WarpShape::kColumn;
          int row = warprow + threadIdx.x / Policy::WarpShape::kColumn +
                    i * Policy::WarpShape::kRow;
          if (aligned_block || tilerow + row < seq_len && col < head_dim) {
            *(access_type_b*)(O + (tilerow + row) * head_dim + col) =
                *(access_type_b*)(&frag_o[k][i * col_size + j]);
          }
        }
      }
    }
  }

  // store partial_m, partial_l, [split_num, seqlen]
  if constexpr (is_split) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = threadIdx.x % Policy::WarpShape::kColumn; i < row_per_lane;
         i += Policy::WarpShape::kColumn) {
      int row = i * Policy::WarpShape::kRow +
                threadIdx.x / Policy::WarpShape::kColumn;
      partial_m[tilerow + warprow + row] = reg_m[i];
      partial_l[tilerow + warprow + row] = reg_l[i];
    }
  }
}

template <bool aligned_block, int max_split_num, int max_head_dim>
__global__ void __launch_bounds__(32)
    flash_attention_combine(float* __restrict__ partial_o,
                            float* __restrict__ partial_m,
                            float* __restrict__ partial_l,
                            float* __restrict__ O, int seq_len, int head_dim,
                            int split_num) {
  O += blockIdx.z * seq_len * head_dim;
  partial_o += blockIdx.z * seq_len * head_dim * split_num;
  partial_m += blockIdx.z * seq_len * split_num;
  partial_l += blockIdx.z * seq_len * split_num;
  constexpr int br = 32;
  constexpr int tr = (32 << 2) / max_head_dim;
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
    int row = wr + (threadIdx.x << 2) / max_head_dim;
    int col = (threadIdx.x << 2) % max_head_dim;
    if (aligned_block && row >= seq_len && col >= head_dim) {
      break;
    }
    float o_acc[4] = {};
    float reg_o[4];
    CUTLASS_PRAGMA_UNROLL
    for (int split_id = 0; split_id < max_split_num; ++split_id) {
      if (split_id >= split_num) {
        break;
      }
      *(float4*)(&reg_o[0]) = *(
          float4*)(&partial_o[split_id * (seq_len * head_dim) +
                              (tilerow + wr) * head_dim + (threadIdx.x << 2)]);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        o_acc[i] += reg_o[i] * s_snum_br[split_id][row];
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      reg_o[i] = (o_acc[i]);
    }
    *(float4*)(&O[(tilerow + wr) * head_dim + (threadIdx.x << 2)]) =
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
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            O.data_ptr<float>(), nullptr, nullptr, seq_len, head_dim, seq_len);
  } else {
    int split_len =
        (((seq_len + BC - 1) / BC) + split_num - 1) / split_num * BC;
    auto partial_o =
        torch::empty({batch, heads, split_num, seq_len, head_dim}, Q.options());
    auto partial_m =
        torch::empty({batch, heads, split_num, seq_len}, Q.options());
    auto partial_l =
        torch::empty({batch, heads, split_num, seq_len}, Q.options());
    flash_attention_kernel<aligned_block, max_head_dim, is_causal, is_split>
        <<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(), seq_len, head_dim, split_len);

    bool aligned_block_combine =
        (seq_len % 32 == 0 && head_dim == max_head_dim);
    INT_DISPATCHER_6(kSplitNum, split_num, 1, 2, 4, 8, 16, 32, [&]() {
      BOOL_DISPATCHER(kAlignedBlock, aligned_block_combine, [&]() {
        flash_attention_combine<kAlignedBlock, kSplitNum, max_head_dim>
            <<<dim3(1, (seq_len + 32 - 1) / 32, batch * heads), dim3(32)>>>(
                partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
                partial_l.data_ptr<float>(), O.data_ptr<float>(), seq_len,
                head_dim, split_num);
      })
    });
  }
}

torch::Tensor flash_attention_simt_forward(const torch::Tensor& Q,
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
  bool aligned = (head_dim % 4 == 0 && seq_len % 4 == 0 &&
                  reinterpret_cast<long long>(Q.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<long long>(K.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<long long>(V.data_ptr()) % 16 == 0);
  // flaot4 load
  TORCH_CHECK(aligned, "must align to 16bit");
  // reduce kernel load o
  TORCH_CHECK(head_dim <= 128, "unsupport head dim > 128");

  auto O = torch::empty_like(Q);
  bool aligned_block =
      (seq_len % BR == 0 && seq_len % BC == 0 && head_dim % 64 == 0);
  bool is_causal = true;
  int split_num = 1;  // get_split_num(batch, heads, seq_len, 32);

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
