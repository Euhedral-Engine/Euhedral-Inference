#pragma once
#include "../numeric.cuh"

namespace q3 {
// Decode one signed 3-bit code (range [-4, 3]) from slot 0 or 1 of a 6-bit field.
// SCOPE: thread-local; pure.
static __device__ __forceinline__ int decode_code(unsigned int pair, unsigned int slot) {
    int code = (pair >> (slot * 3)) & 7;
    code -= (code & 4) ? 8 : 0;
    return code;
}

// Two orders exist and are separate numerical contracts; do not interchange them.

// Dequantized weight value code * scale. Exact in FP32 (at most 14 significant
// bits: FP16 mantissa times a 3-bit code). Used when a weight is staged for MMA.
static __device__ __forceinline__ float apply_scale(int code, float scale) {
    return (float)code * scale;
}

// Activation contribution in the scalar reference's FP32 order: (x * code) * scale.
// SIMT paths use this so their per-stripe sums match the reference bit for bit.
static __device__ __forceinline__ float scaled_product(float x, int code, float scale) {
    return x * (float)code * scale;
}


}  // namespace q3
