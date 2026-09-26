#pragma once
#include "../primitives/packed_load.cuh"
#include "../primitives/decode.cuh"
#include "../primitives/accumulation.cuh"
#include "../primitives/writeback.cuh"

namespace q3 {
// Scalar reference: one CTA per (row, output); threads stride K. Kept independent
// of every shared-load primitive so it remains an oracle for the tiled paths.
static __device__ __forceinline__ void scalar_reference(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset, float* partial) {
    unsigned int linear = blockIdx.x;
    unsigned int row = linear / out_features, out = linear % out_features;
    if (row >= rows) return;
    const Layout w(weights, in_features, scale_offset);
    float sum = 0.0f;
    for (unsigned int k = threadIdx.x; k < in_features; k += blockDim.x) {
        unsigned long long g = w.group(out, k / kGroup);
        int code = load_code_at(w.group_bytes(g), k % kGroup);
        // The reference has always formed its scale index in 32-bit arithmetic, which
        // wraps once a single tensor reaches 2^32 groups (>= 104 GiB). Preserved as-is;
        // the tiled paths use the 64-bit group index.
        unsigned int scale_index = out * w.groups + k / kGroup;
        sum = sum + scaled_product(bf16_to_float(input[(unsigned long long)row * in_features + k]), code,
                load_scale_at(w, scale_index));
    }
    reduce_cta_128(partial, sum);
    if (threadIdx.x == 0) write_bf16(output, row, out, out_features, partial[0]);
}


}  // namespace q3
