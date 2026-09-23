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

    /// Executes one row-split Q4 projection over BF16 activations.
    default void linearQ4Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        throw new UnsupportedOperationException("Q4 linear is not implemented by this GPU");
    }

    /// Executes one row-split Q5 projection over BF16 activations.
    default void linearQ5Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        throw new UnsupportedOperationException("Q5 linear is not implemented by this GPU");
    }

    /// Multiplies BF16 activations and weights, preserving FP32 projection results.
    default void linearBf16ToFloat(
            long inputAddress, long weightsAddress, long outputAddress, int rows, int inFeatures, int outFeatures) {
        throw new UnsupportedOperationException("BF16-to-FP32 linear is not implemented by this GPU");
    }

    default void gdnControlFp32(
            long aProjectionAddress,
            long bProjectionAddress,
            long aLogAddress,
            long dtBiasAddress,
            long gOutputAddress,
            long betaOutputAddress,
            int rows,
            int heads) {
        throw new UnsupportedOperationException("GDN control operation is not implemented by this GPU");
    }

    default void gdnConvolutionBf16(
            long queryKeyAddress,
            long valueZAddress,
            long convolutionWeightsAddress,
            long convolutionStateAddress,
            long outputAddress,
            int rows,
            int queryKeyWidth,
            int valueWidth,
            int convolutionWidth,
            int kernelSize) {
        throw new UnsupportedOperationException("GDN convolution is not implemented by this GPU");
    }

    default void gdnRecurrenceBf16(
            long convolvedAddress,
            long gAddress,
            long betaAddress,
            long recurrentStateAddress,
            long outputAddress,
            int rows,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            float outputScale) {
        throw new UnsupportedOperationException("GDN recurrence is not implemented by this GPU");
    }

    default void gdnGatedRmsNormBf16(
            long recurrentAddress,
            long valueZAddress,
            long normWeightAddress,
            long outputAddress,
            int rows,
            int valueHeads,
            int headDim,
            float epsilon) {
        throw new UnsupportedOperationException("GDN gated RMSNorm is not implemented by this GPU");
    }

    default void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {
        throw new UnsupportedOperationException("BF16 residual add is not implemented by this GPU");
    }

    default void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {
        throw new UnsupportedOperationException("BF16 SwiGLU is not implemented by this GPU");
    }

    default void zeroDeviceMemory(long address, long byteSize) {
        throw new UnsupportedOperationException("device memory zeroing is not implemented by this GPU");
    }
}
