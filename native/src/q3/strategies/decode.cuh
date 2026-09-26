#pragma once
#include "../primitives/activation.cuh"
#include "../primitives/packed_load.cuh"
#include "../primitives/decode.cuh"
#include "../primitives/accumulation.cuh"
#include "../primitives/writeback.cuh"

namespace q3 {
// Cooperative decode for R <= 4 token rows. Each of 4 warps owns 2 output
// columns; a CTA covers 8 columns x R rows. Per 128-wide K step a lane holds four
// K-stripe activations per row and accumulates four FP32 stripes per (row, col),
// reusing each loaded activation across both columns.
template<int R>
static __device__ __forceinline__ void cooperative_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    constexpr int kWarpCols = 2, kStripes = 4;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + 7u) / 8u;
    const unsigned int first_out = (blockIdx.x % output_tiles) * 8 + warp * kWarpCols;
    const unsigned int first_row = (blockIdx.x / output_tiles) * R;
    const Layout w(weights, in_features, scale_offset);
    float sums[R][kWarpCols][kStripes] = {};
    for (unsigned int k = 0; k < in_features; k += 2 * kGroup) {
        float x[R][kStripes];
        load_activation_stripes<R, kStripes>(x, input, rows, in_features, first_row, k, lane);
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            if (first_out + n < out_features) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    unsigned long long g = w.group(first_out + n, k / kGroup + half);
                    unsigned int pairs = load_packed_pair(w, g, lane);
                    float scale = load_group_scale(w, g, lane);
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        int code = decode_code(stripe_pair(pairs, lane, p), lane & 1);
                        #pragma unroll
                        for (int r = 0; r < R; r++)
                            sums[r][n][half * 2 + p] += scaled_product(x[r][half * 2 + p], code, scale);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            float sum = reduce_stripes(sums[r][n]);
            if (lane == 0 && first_row + r < rows && first_out + n < out_features)
                write_bf16(output, first_row + r, first_out + n, out_features, sum);
        }
    }
}


}  // namespace q3
