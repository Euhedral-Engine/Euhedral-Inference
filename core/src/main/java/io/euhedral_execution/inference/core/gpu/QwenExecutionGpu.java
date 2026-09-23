package io.euhedral_execution.inference.core.gpu;

/// Synchronous GPU operations used by the current Qwen instruction slice.
public interface QwenExecutionGpu extends GpuMemory {

    void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize);

    void synchronize();

    default void rmsNormBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("BF16 RMS norm is not implemented by this GPU");
    }

    default void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        throw new UnsupportedOperationException("Q3 linear is not implemented by this GPU");
    }
}
