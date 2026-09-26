#pragma once
#include "../numeric.cuh"
#include "mma.cuh"

namespace q3 {
// Write one FP32 result as BF16 (round-to-nearest-even). Bounds are the caller's.
static __device__ __forceinline__ void write_bf16(
        unsigned short* output, unsigned int row, unsigned int col, unsigned int out_features, float value) {
    output[(unsigned long long)row * out_features + col] = float_to_bf16(value);
}

// Store this warp's accumulators to a borrowed shared FP32 tile (row-major,
// leading dimension Tile::kCols). SCOPE: warp-collective; no barrier.
template<class Tile>
static __device__ __forceinline__ void store_accumulators(
        float* result, Accumulators<Tile::kFrags>& acc, unsigned int warp) {
    #pragma unroll
    for (int m = 0; m < Tile::kFrags; m++)
        wmma::store_matrix_sync(result + Tile::row(warp, m) * Tile::kCols + Tile::col(warp),
                acc.frag[m], Tile::kCols, wmma::mem_row_major);
}

// Write a ROWS x COLS FP32 shared tile to global BF16, skipping out-of-matrix
// elements. SCOPE: CTA-cooperative; the strategy barriers after the stores above.
template<int ROWS, int COLS, int THREADS>
static __device__ __forceinline__ void write_output_tile(
        unsigned short* output, const float* result, unsigned int rows, unsigned int out_features,
        unsigned int row_start, unsigned int out_start, unsigned int thread) {
    for (unsigned int i = thread; i < ROWS * COLS; i += THREADS) {
        unsigned int r = row_start + i / COLS, n = out_start + i % COLS;
        if (r < rows && n < out_features) write_bf16(output, r, n, out_features, result[i]);
    }
}


}  // namespace q3
