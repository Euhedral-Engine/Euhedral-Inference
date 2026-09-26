#pragma once
#include "../primitives/activation.cuh"
#include "../primitives/staging.cuh"
#include "../primitives/mma.cuh"
#include "../primitives/writeback.cuh"

namespace q3 {
// Tiled WMMA prefill. The CTA owns Tile::kRows x Tile::kCols outputs and consumes
// K one G64 group at a time:
//   stage A (activations) + stage B (decoded hi/lo weights)  -> barrier
//   every warp consumes the staged tile into FP32 accumulators -> barrier
// then stores accumulators to shared, barriers, and writes BF16 output. The two
// per-step barriers are the single-buffer reuse edges: staged tiles are visible
// only after the first, and may be overwritten only after the second.
template<class Tile>
static __device__ __forceinline__ void tiled_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, __nv_bfloat16* a, __nv_bfloat16* b_hi, __nv_bfloat16* b_lo,
        float* result) {
    constexpr int kThreads = Tile::kWarps * 32;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + Tile::kCols - 1u) / Tile::kCols;
    const unsigned int row_start = (blockIdx.x / output_tiles) * Tile::kRows;
    const unsigned int out_start = (blockIdx.x % output_tiles) * Tile::kCols;
    const Layout w(weights, in_features, scale_offset);
    Accumulators<Tile::kFrags> acc;
    acc.fill();
    for (unsigned int base = 0; base < in_features; base += kGroup) {
        stage_activation_tile<Tile::kRows, kGroup, kThreads>(
                a, input, rows, in_features, row_start, base, threadIdx.x);
        stage_weight_tile<Tile::kCols, Tile::kWarps>(b_hi, b_lo, w, out_start, out_features, base, warp, lane);
        __syncthreads();
        consume_mma_tile<Tile, kGroup, kGroup, kGroup>(acc, a, b_hi, b_lo, warp);
        __syncthreads();
    }
    store_accumulators<Tile>(result, acc, warp);
    __syncthreads();
    write_output_tile<Tile::kRows, Tile::kCols, kThreads>(
            output, result, rows, out_features, row_start, out_start, threadIdx.x);
}

// Tile geometries used by the current routes. 2 x 2 warps with one or two
// fragments per warp give 32 x 32 and 64 x 32 CTA tiles. Host dispatch in
// q3_linear_bf16.c sizes grids with the same constants.
using Prefill32 = WarpTile<2, 2, 1>;
using Prefill64 = WarpTile<2, 2, 2>;


}  // namespace q3
