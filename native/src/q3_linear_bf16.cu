#include <cuda_bf16.h>
#include <mma.h>

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

// Six cooperative 32-bit reads cover a G64 group. Each lane owns two codes;
// shuffles also share the single FP16 scale instead of reloading it per value.
static __device__ __forceinline__ unsigned int q3_pair(
        const unsigned char* weights, unsigned long long group, unsigned int lane) {
    unsigned int word = lane < 6 ? ((const unsigned int*)(weights + group * 24ull))[lane] : 0;
    unsigned int bit = lane * 6u;
    unsigned int lo = __shfl_sync(0xffffffffu, word, bit >> 5);
    unsigned int hi = __shfl_sync(0xffffffffu, word, (bit >> 5) + 1);
    return (unsigned int)((((unsigned long long)hi << 32) | lo) >> (bit & 31u)) & 63u;
}

template<int R>
static __device__ __forceinline__ void q3_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + 7u) / 8u;
    const unsigned int first_out = (blockIdx.x % output_tiles) * 8 + warp * 2;
    const unsigned int first_row = (blockIdx.x / output_tiles) * R;
    const unsigned int groups = ((in_features + 127u) / 128u) * 2u;
    const unsigned short* scales = (const unsigned short*)(weights + scale_offset);
    // Four logical K stripes per lane preserve the scalar reference's FP32
    // summation order, while one warp still owns two output columns and reuses X.
    float sums[R][2][4] = {};
    for (unsigned int k = 0; k < in_features; k += 128) {
        float x[R][4];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                unsigned int column = k + lane + p * 32;
                unsigned long long offset = (unsigned long long)(first_row + r) * in_features + column;
                x[r][p] = first_row + r < rows && column < in_features ? bf16_to_float(input[offset]) : 0;
            }
        }
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (first_out + n < out_features) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    unsigned long long g = (unsigned long long)(first_out + n) * groups + k / 64 + half;
                    unsigned int pairs = q3_pair(weights, g, lane);
                    float scale = fp16_to_float(__shfl_sync(0xffffffffu, lane == 0 ? (unsigned int)scales[g] : 0u, 0));
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        unsigned int pair = __shfl_sync(0xffffffffu, pairs, lane / 2 + p * 16);
                        int code = (pair >> ((lane & 1) * 3)) & 7;
                        code -= (code & 4) ? 8 : 0;
                        #pragma unroll
                        for (int r = 0; r < R; r++)
                            sums[r][n][half * 2 + p] += x[r][half * 2 + p] * (float)code * scale;
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            float sum = (sums[r][n][0] + sums[r][n][2]) + (sums[r][n][1] + sums[r][n][3]);
            #pragma unroll
            for (int distance = 16; distance; distance >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, distance);
            if (lane == 0 && first_row + r < rows && first_out + n < out_features)
                output[(unsigned long long)(first_row + r) * out_features + first_out + n] = float_to_bf16(sum);
        }
    }
}

#define DECODE_KERNEL(R) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_##R( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    q3_decode<R>(input, weights, output, rows, in_features, out_features, scale_offset); \
}
DECODE_KERNEL(1)
DECODE_KERNEL(2)
DECODE_KERNEL(4)

// A CTA computes 32 token rows x 32 output columns, consuming K in G64 tiles.
// Both activation and decoded weight tiles are reused by four WMMA warps.
// Two BF16 weight components preserve the FP16-scale times Q3-code value, rather
// than silently rounding the weight to BF16. All accumulators remain FP32.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 a[32 * 64];
    __shared__ __align__(32) __nv_bfloat16 b_hi[32 * 64];
    __shared__ __align__(32) __nv_bfloat16 b_lo[32 * 64];
    __shared__ __align__(32) float result[32 * 32];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int output_tiles = (out_features + 31u) / 32u;
    unsigned int row_start = (blockIdx.x / output_tiles) * 32, out_start = (blockIdx.x % output_tiles) * 32;
    unsigned int groups = ((in_features + 127u) / 128u) * 2u;
    const unsigned short* scales = (const unsigned short*)(weights + scale_offset);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (unsigned int base = 0; base < in_features; base += 64) {
        for (unsigned int i = threadIdx.x; i < 32 * 64; i += 128) {
            unsigned int r = row_start + i / 64, k = base + i % 64;
            a[i] = __float2bfloat16(r < rows && k < in_features ? bf16_to_float(input[(unsigned long long)r * in_features + k]) : 0.0f);
        }
        for (unsigned int col = warp; col < 32; col += 4) {
            unsigned int codes = 0;
            float scale = 0.0f;
            if (out_start + col < out_features) {
                unsigned long long g = (unsigned long long)(out_start + col) * groups + base / 64;
                codes = q3_pair(weights, g, lane);
                scale = fp16_to_float(__shfl_sync(0xffffffffu, lane == 0 ? (unsigned int)scales[g] : 0u, 0));
            }
            #pragma unroll
            for (int p = 0; p < 2; p++) {
                int code = (codes >> (p * 3)) & 7; code -= (code & 4) ? 8 : 0;
                float weight = (float)code * scale;
                unsigned int i = col * 64 + lane * 2 + p;
                b_hi[i] = __float2bfloat16(weight);
                b_lo[i] = __float2bfloat16(weight - __bfloat162float(b_hi[i]));
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a + (warp / 2) * 16 * 64 + k, 64);
            wmma::load_matrix_sync(bf, b_hi + (warp % 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc, af, bf, acc);
            wmma::load_matrix_sync(bf, b_lo + (warp % 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(result + (warp / 2) * 16 * 32 + (warp % 2) * 16, acc, 32, wmma::mem_row_major);
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < 32 * 32; i += 128) {
        unsigned int r = row_start + i / 32, n = out_start + i % 32;
        if (r < rows && n < out_features) output[(unsigned long long)r * out_features + n] = float_to_bf16(result[i]);
    }
}

// Double the row tile for the large Q3 MLP projections. Each warp holds two
// FP32 accumulators, so a packed/dequantized weight tile is reused across 64
// token rows rather than 32. The 32-row kernel above remains the fallback.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_prefill_64(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 a[64 * 64];
    __shared__ __align__(32) __nv_bfloat16 b_hi[32 * 64];
    __shared__ __align__(32) __nv_bfloat16 b_lo[32 * 64];
    __shared__ __align__(32) float result[64 * 32];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int output_tiles = (out_features + 31u) / 32u;
    unsigned int row_start = (blockIdx.x / output_tiles) * 64, out_start = (blockIdx.x % output_tiles) * 32;
    unsigned int groups = ((in_features + 127u) / 128u) * 2u;
    const unsigned short* scales = (const unsigned short*)(weights + scale_offset);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    for (unsigned int base = 0; base < in_features; base += 64) {
        for (unsigned int i = threadIdx.x; i < 64 * 64; i += 128) {
            unsigned int r = row_start + i / 64, k = base + i % 64;
            a[i] = __float2bfloat16(r < rows && k < in_features
                    ? bf16_to_float(input[(unsigned long long)r * in_features + k]) : 0.0f);
        }
        for (unsigned int col = warp; col < 32; col += 4) {
            unsigned int codes = 0;
            float scale = 0.0f;
            if (out_start + col < out_features) {
                unsigned long long g = (unsigned long long)(out_start + col) * groups + base / 64;
                codes = q3_pair(weights, g, lane);
                scale = fp16_to_float(__shfl_sync(0xffffffffu, lane == 0 ? (unsigned int)scales[g] : 0u, 0));
            }
            #pragma unroll
            for (int p = 0; p < 2; p++) {
                int code = (codes >> (p * 3)) & 7; code -= (code & 4) ? 8 : 0;
                float weight = (float)code * scale;
                unsigned int i = col * 64 + lane * 2 + p;
                b_hi[i] = __float2bfloat16(weight);
                b_lo[i] = __float2bfloat16(weight - __bfloat162float(b_hi[i]));
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af0, af1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af0, a + (warp / 2) * 16 * 64 + k, 64);
            wmma::load_matrix_sync(af1, a + ((warp / 2) * 16 + 32) * 64 + k, 64);
            wmma::load_matrix_sync(bf, b_hi + (warp % 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc0, af0, bf, acc0);
            wmma::mma_sync(acc1, af1, bf, acc1);
            wmma::load_matrix_sync(bf, b_lo + (warp % 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc0, af0, bf, acc0);
            wmma::mma_sync(acc1, af1, bf, acc1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(result + (warp / 2) * 16 * 32 + (warp % 2) * 16,
            acc0, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(result + ((warp / 2) * 16 + 32) * 32 + (warp % 2) * 16,
            acc1, 32, wmma::mem_row_major);
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < 64 * 32; i += 128) {
        unsigned int r = row_start + i / 32, n = out_start + i % 32;
        if (r < rows && n < out_features) output[(unsigned long long)r * out_features + n] = float_to_bf16(result[i]);
    }
}
