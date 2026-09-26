#pragma once
#include <cuda_bf16.h>

namespace q3 {
static __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float((unsigned int)value << 16);
}
static __device__ __forceinline__ float fp16_to_float(unsigned short value) {
    unsigned int sign = ((unsigned int)value & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)value >> 10) & 31u;
    unsigned int mantissa = (unsigned int)value & 1023u;
    if (exponent == 0u) {
        if (mantissa == 0u) return __uint_as_float(sign);
        int shift = 0;
        while ((mantissa & 1024u) == 0u) { mantissa <<= 1; --shift; }
        mantissa &= 1023u;
        return __uint_as_float(sign | ((unsigned int)(113 + shift) << 23) | (mantissa << 13));
    }
    if (exponent == 31u) return __uint_as_float(sign | 0x7f800000u | (mantissa << 13));
    return __uint_as_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}
static __device__ __forceinline__ unsigned short float_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (unsigned short)(bits >> 16);
}


}  // namespace q3
