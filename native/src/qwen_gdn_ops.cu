#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;


__device__ __forceinline__ float qwen_gdn_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float qwen_gdn_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float qwen_gdn_silu(float value) {
    return value * qwen_gdn_sigmoid(value);
}

extern "C" __global__ void euhedral_gdn_control_fp32(
        const float* aProjection, const float* bProjection, const float* aLog, const float* dtBias,
        float* gOutput, float* betaOutput, uint32_t rows, uint32_t heads) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<uint64_t>(rows) * heads) return;
    const uint32_t head = static_cast<uint32_t>(index % heads);
    const float shifted = aProjection[index] + dtBias[head];
    const float softplus = fmaxf(shifted, 0.0f) + log1pf(expf(-fabsf(shifted)));
    gOutput[index] = -expf(aLog[head]) * softplus;
    betaOutput[index] = qwen_gdn_sigmoid(bProjection[index]);
}

extern "C" __global__ void euhedral_gdn_convolution_bf16(
        const __nv_bfloat16* queryKey, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* convolutionWeights, __nv_bfloat16* convolutionState,
        __nv_bfloat16* output, uint32_t rows, uint32_t queryKeyWidth,
        uint32_t valueWidth, uint32_t convolutionWidth, uint32_t kernelSize) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= convolutionWidth) return;
    const uint32_t historyWidth = kernelSize - 1;
    float history[31];
    for (uint32_t i = 0; i < historyWidth; i++) {
        history[i] = __bfloat162float(convolutionState[static_cast<uint64_t>(channel) * historyWidth + i]);
    }
    for (uint32_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernelSize; tap++) {
            float value;
            if (tap < historyWidth) {
                value = history[tap];
            } else if (channel < queryKeyWidth) {
                value = __bfloat162float(queryKey[static_cast<uint64_t>(row) * queryKeyWidth + channel]);
            } else {
                value = __bfloat162float(valueZ[static_cast<uint64_t>(row) * valueWidth * 2 + channel - queryKeyWidth]);
            }
            sum = fmaf(value, __bfloat162float(convolutionWeights[static_cast<uint64_t>(tap) * convolutionWidth + channel]), sum);
        }
        output[static_cast<uint64_t>(row) * convolutionWidth + channel] = __float2bfloat16_rn(qwen_gdn_silu(sum));
        if (historyWidth > 0) {
            for (uint32_t i = 0; i + 1 < historyWidth; i++) history[i] = history[i + 1];
            const float current = channel < queryKeyWidth
                    ? __bfloat162float(queryKey[static_cast<uint64_t>(row) * queryKeyWidth + channel])
                    : __bfloat162float(valueZ[static_cast<uint64_t>(row) * valueWidth * 2 + channel - queryKeyWidth]);
            history[historyWidth - 1] = current;
        }
    }
    for (uint32_t i = 0; i < historyWidth; i++) {
        convolutionState[static_cast<uint64_t>(channel) * historyWidth + i] = __float2bfloat16_rn(history[i]);
    }
}

extern "C" __global__ void euhedral_gdn_recurrence_bf16(
        const __nv_bfloat16* convolved, const float* g, const float* beta,
        float* recurrentState, __nv_bfloat16* output, uint32_t rows,
        uint32_t keyHeads, uint32_t valueHeads, uint32_t keyHeadDim,
        uint32_t valueHeadDim, float outputScale) {
    const uint32_t valueRow = blockIdx.x;
    const uint32_t valueHead = valueRow / valueHeadDim;
    const uint32_t valueColumn = valueRow % valueHeadDim;
    if (valueHead >= valueHeads || blockDim.x != keyHeadDim) return;
    const uint32_t queryKeyWidth = 2 * keyHeads * keyHeadDim;
    const uint32_t convolvedWidth = queryKeyWidth + valueHeads * valueHeadDim;
    const uint32_t keyHead = valueHead / (valueHeads / keyHeads);
    const uint32_t lane = threadIdx.x;
    const uint64_t stateOffset = (static_cast<uint64_t>(valueHead) * valueHeadDim + valueColumn) * keyHeadDim;
    float stateValue = recurrentState[stateOffset + lane];
    __shared__ float scratch[128];

    for (uint32_t row = 0; row < rows; row++) {
        const uint64_t rowOffset = static_cast<uint64_t>(row) * convolvedWidth;
        const uint32_t queryBase = keyHead * keyHeadDim;
        const uint32_t keyBase = keyHeads * keyHeadDim + queryBase;
        const uint32_t valueBase = queryKeyWidth + valueHead * valueHeadDim + valueColumn;
        const float query = __bfloat162float(convolved[rowOffset + queryBase + lane]);
        const float key = __bfloat162float(convolved[rowOffset + keyBase + lane]);
        const float normalizedKey = key * rsqrtf(qwen_gdn_reduce_sum(key * key, scratch) + 1.0e-6f);
        const float normalizedQuery = query * rsqrtf(qwen_gdn_reduce_sum(query * query, scratch) + 1.0e-6f);
        const float stateKeyDot = qwen_gdn_reduce_sum(stateValue * normalizedKey, scratch);
        const float alpha = expf(g[static_cast<uint64_t>(row) * valueHeads + valueHead]);
        const float delta = beta[static_cast<uint64_t>(row) * valueHeads + valueHead]
                * (__bfloat162float(convolved[rowOffset + valueBase]) - alpha * stateKeyDot);
        stateValue = alpha * stateValue + delta * normalizedKey;
        const float outputSum = qwen_gdn_reduce_sum(stateValue * normalizedQuery, scratch);
        if (lane == 0) output[static_cast<uint64_t>(row) * valueHeads * valueHeadDim + valueRow]
                = __float2bfloat16_rn(outputSum * outputScale);
    }
    recurrentState[stateOffset + lane] = stateValue;
}

extern "C" __global__ void euhedral_gdn_gated_rms_norm_bf16(
        const __nv_bfloat16* recurrent, const __nv_bfloat16* valueZ,
        const __nv_bfloat16* normWeight, __nv_bfloat16* output,
        uint32_t rows, uint32_t valueHeads, uint32_t headDim, float epsilon) {
    const uint32_t rowHead = blockIdx.x;
    const uint32_t row = rowHead / valueHeads;
    const uint32_t head = rowHead % valueHeads;
    if (row >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t width = valueHeads * headDim;
    const uint64_t recurrentOffset = static_cast<uint64_t>(row) * width + static_cast<uint64_t>(head) * headDim;
    const uint64_t zOffset = static_cast<uint64_t>(row) * width * 2 + width + static_cast<uint64_t>(head) * headDim;
    float partial = 0.0f;
    for (uint32_t i = lane; i < headDim; i += blockDim.x) {
        const float value = __bfloat162float(recurrent[recurrentOffset + i]);
        partial = fmaf(value, value, partial);
    }
    __shared__ float scratch[128];
    const float sum = qwen_gdn_reduce_sum(partial, scratch);
    const float inverse = rsqrtf(sum / static_cast<float>(headDim) + epsilon);
    for (uint32_t i = lane; i < headDim; i += blockDim.x) {
        const float x = __bfloat162float(recurrent[recurrentOffset + i]);
        const float z = __bfloat162float(valueZ[zOffset + i]);
        output[recurrentOffset + i] = __float2bfloat16_rn(x * inverse * __bfloat162float(normWeight[i]) * qwen_gdn_silu(z));
    }
}
