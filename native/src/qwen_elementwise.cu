#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


__device__ __forceinline__ float euhedral_silu(float value) {
    return value / (1.0f + expf(-value));
}

extern "C" __global__ void euhedral_residual_add_bf16(
        const __nv_bfloat16* residual, const __nv_bfloat16* delta,
        __nv_bfloat16* output, uint32_t elementCount) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elementCount) {
        output[index] = __float2bfloat16_rn(
                __bfloat162float(residual[index]) + __bfloat162float(delta[index]));
    }
}

extern "C" __global__ void euhedral_swiglu_bf16(
        const __nv_bfloat16* gateUp, __nv_bfloat16* output,
        uint32_t rows, uint32_t intermediateSize) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * intermediateSize;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index / intermediateSize);
    const uint32_t column = static_cast<uint32_t>(index % intermediateSize);
    const uint64_t rowOffset = static_cast<uint64_t>(row) * intermediateSize * 2;
    const float gate = __bfloat162float(gateUp[rowOffset + column]);
    const float up = __bfloat162float(gateUp[rowOffset + intermediateSize + column]);
    output[index] = __float2bfloat16_rn(euhedral_silu(gate) * up);
}
