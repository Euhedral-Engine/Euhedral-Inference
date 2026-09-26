#pragma once
#include <mma.h>

namespace q3 {
namespace wmma = nvcuda::wmma;
using AccumulatorFragment = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// Warp placement inside a CTA output tile. WARPS_M x WARPS_N warps each own
// M_FRAGS vertically strided 16x16 fragments in a single 16-column band. The CTA
// tile is (16 * WARPS_M * M_FRAGS) x (16 * WARPS_N).
template<int WARPS_M, int WARPS_N, int M_FRAGS>
struct WarpTile {
    static constexpr int kWarps = WARPS_M * WARPS_N;
    static constexpr int kRows = 16 * WARPS_M * M_FRAGS;
    static constexpr int kCols = 16 * WARPS_N;
    static constexpr int kFrags = M_FRAGS;
    static __device__ __forceinline__ unsigned int row(unsigned int warp, int m) {
        return (warp / WARPS_N) * 16 + m * 16 * WARPS_M;
    }
    static __device__ __forceinline__ unsigned int col(unsigned int warp) { return (warp % WARPS_N) * 16; }
};

// FP32 accumulators owned by one warp for the whole K loop. Only fill() and
// consume_mma_tile() modify them; they become final after the last K step.
template<int M_FRAGS>
struct Accumulators {
    AccumulatorFragment frag[M_FRAGS];
    __device__ __forceinline__ void fill() {
        #pragma unroll
        for (int m = 0; m < M_FRAGS; m++) wmma::fill_fragment(frag[m], 0.0f);
    }
};

// Consume one staged K tile: acc += A * (B_hi + B_lo) for this warp's fragments.
// SCOPE: warp-collective (WMMA). BORROWS: the shared A and B tiles, read-only; the
// strategy must not re-stage them until every warp has returned and barriered.
// Each B fragment is loaded once and reused across all M_FRAGS accumulators.
template<class Tile, int K_TILE, int LDA, int LDB>
static __device__ __forceinline__ void consume_mma_tile(
        Accumulators<Tile::kFrags>& acc, const __nv_bfloat16* a, const __nv_bfloat16* b_hi,
        const __nv_bfloat16* b_lo, unsigned int warp) {
    #pragma unroll
    for (unsigned int k = 0; k < K_TILE; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[Tile::kFrags];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) wmma::load_matrix_sync(af[m], a + Tile::row(warp, m) * LDA + k, LDA);
        wmma::load_matrix_sync(bf, b_hi + Tile::col(warp) * LDB + k, LDB);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) wmma::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
        wmma::load_matrix_sync(bf, b_lo + Tile::col(warp) * LDB + k, LDB);
        #pragma unroll
        for (int m = 0; m < Tile::kFrags; m++) wmma::mma_sync(acc.frag[m], af[m], bf, acc.frag[m]);
    }
}


}  // namespace q3
