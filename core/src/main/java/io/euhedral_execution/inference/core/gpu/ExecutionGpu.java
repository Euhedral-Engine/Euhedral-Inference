package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/// Synchronous GPU operations used by the current Qwen instruction slice.
public abstract class ExecutionGpu implements GpuMemory {

    protected static final FunctionDescriptor MALLOC =
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor FREE = FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
    protected static final FunctionDescriptor DEVICE_MEMORY_INFO =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
    protected static final FunctionDescriptor COPY = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor EMBED_Q3 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor SYNCHRONIZE = FunctionDescriptor.of(ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor RMS_NORM_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT);
    protected static final FunctionDescriptor RMS_NORM_UNIT_OFFSET_BF16 = RMS_NORM_BF16;
    protected static final FunctionDescriptor LINEAR_Q3_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG);
    protected static final FunctionDescriptor LINEAR_QUANTIZED_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor LINEAR_BF16_TO_FLOAT = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor GDN_CONTROL_FP32 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor GDN_CONVOLUTION_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor GDN_RECURRENCE_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT);
    protected static final FunctionDescriptor GDN_GATED_RMS_NORM_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_FLOAT);
    protected static final FunctionDescriptor RESIDUAL_ADD_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor SWIGLU_BF16 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);
    protected static final FunctionDescriptor ZERO_DEVICE_MEMORY =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    protected static final int CUDA_FORMAT_MISMATCH = -3;

    protected static MethodHandle bind(
            Linker linker, SymbolLookup symbols, String name, FunctionDescriptor descriptor) {
        MemorySegment symbol =
                symbols.find(name).orElseThrow(() -> new GpuMemoryException("native symbol not found: " + name));
        return linker.downcallHandle(symbol, descriptor);
    }

    public abstract void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize);

    public abstract void synchronize();

    public void rmsNormBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("BF16 RMS norm is not implemented by this GPU");
    }

    /// Applies Qwen's unit-offset RMSNorm convention: normalized * (1 + weight).
    public void rmsNormUnitOffsetBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        throw new UnsupportedOperationException("unit-offset BF16 RMS norm is not implemented by this GPU");
    }

    public void linearQ3Bf16(
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
    public void linearQ4Bf16(
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
    public void linearQ5Bf16(
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
    public void linearBf16ToFloat(
            long inputAddress, long weightsAddress, long outputAddress, int rows, int inFeatures, int outFeatures) {
        throw new UnsupportedOperationException("BF16-to-FP32 linear is not implemented by this GPU");
    }

    public void gdnControlFp32(
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

    public void gdnConvolutionBf16(
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

    public void gdnRecurrenceBf16(
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

    public void gdnGatedRmsNormBf16(
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

    public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {
        throw new UnsupportedOperationException("BF16 residual add is not implemented by this GPU");
    }

    public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {
        throw new UnsupportedOperationException("BF16 SwiGLU is not implemented by this GPU");
    }

    public void zeroDeviceMemory(long address, long byteSize) {
        throw new UnsupportedOperationException("device memory zeroing is not implemented by this GPU");
    }
}
