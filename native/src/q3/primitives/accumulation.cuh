#pragma once

namespace q3 {
// Combine a lane's four K-stripe FP32 sums, then reduce across the warp. The
// pairing ((s0 + s2) + (s1 + s3)) and shuffle tree match the scalar reference.
// SCOPE: warp-collective. PRODUCES: the full sum in lane 0.
static __device__ __forceinline__ float reduce_stripes(const float (&s)[4]) {
    float sum = (s[0] + s[2]) + (s[1] + s[3]);
    #pragma unroll
    for (int distance = 16; distance; distance >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, distance);
    return sum;
}

// CTA tree reduction of one FP32 value per thread for 128 threads.
// SCOPE: CTA-collective, including its barriers. BORROWS: `partial[128]` until
// return. PRODUCES: the total in partial[0], visible to every thread.
static __device__ __forceinline__ void reduce_cta_128(float* partial, float value) {
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
}


}  // namespace q3
