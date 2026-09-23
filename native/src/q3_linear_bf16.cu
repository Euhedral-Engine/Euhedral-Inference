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
static __device__ __forceinline__ int q3_code(const unsigned char* base, unsigned int index) {
    unsigned int bit = index * 3u;
    unsigned int word = (unsigned int)base[bit >> 3];
    word |= (unsigned int)base[(bit >> 3) + 1u] << 8;
    return (int)((word >> (bit & 7u)) & 7u) - (((word >> (bit & 7u)) & 7u) >= 4u ? 8 : 0);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_linear_bf16(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    unsigned int linear = blockIdx.x;
    unsigned int row = linear / out_features, out = linear % out_features;
    if (row >= rows) return;
    unsigned int groups = ((in_features + 127u) / 128u) * 2u;
    const unsigned char* row_weights = weights + (unsigned long long)out * groups * 24ull;
    const unsigned short* scales = (const unsigned short*)(weights + scale_offset);
    float sum = 0.0f;
    for (unsigned int k = threadIdx.x; k < in_features; k += blockDim.x) {
        unsigned int group = k / 64u;
        int code = q3_code(row_weights + group * 24u, k % 64u);
        float scale = fp16_to_float(scales[out * groups + group]);
        sum += bf16_to_float(input[(unsigned long long)row * in_features + k]) * (float)code * scale;
    }
    __shared__ float partial[128];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[(unsigned long long)row * out_features + out] = float_to_bf16(partial[0]);
}
