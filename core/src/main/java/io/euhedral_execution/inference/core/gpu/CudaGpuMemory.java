package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Objects;

/// FFM binding for the stable Euhedral CUDA C ABI.
///
/// CUDA device addresses remain opaque longs. They are converted to zero-size address segments only
/// inside the native calls and are never exposed as dereferenceable Java memory.
public final class CudaGpuMemory extends ExecutionGpu implements AutoCloseable {

    private final Arena arena;
    private final MethodHandle malloc;
    private final MethodHandle free;
    private final MethodHandle deviceMemoryInfo;
    private final MethodHandle copyHostToDevice;
    private final MethodHandle copyDeviceToHost;
    private final MethodHandle embedQ3;
    private final MethodHandle synchronize;
    private final MethodHandle rmsNormBf16;
    private final MethodHandle rmsNormUnitOffsetBf16;
    private final MethodHandle linearQ3Bf16;
    private final MethodHandle linearQuantizedBf16;
    private final MethodHandle linearBf16ToFloat;
    private final MethodHandle gdnControlFp32;
    private final MethodHandle gdnConvolutionBf16;
    private final MethodHandle gdnRecurrenceBf16;
    private final MethodHandle gdnGatedRmsNormBf16;
    private final MethodHandle residualAddBf16;
    private final MethodHandle swiGluBf16;
    private final MethodHandle zeroDeviceMemory;
    private boolean closed;

    public CudaGpuMemory(Path libraryPath) {
        Objects.requireNonNull(libraryPath, "libraryPath");
        Arena loadedLibraryArena = Arena.ofShared();
        try {
            SymbolLookup symbols = SymbolLookup.libraryLookup(libraryPath, loadedLibraryArena);
            Linker linker = Linker.nativeLinker();
            this.arena = loadedLibraryArena;
            this.malloc = bind(linker, symbols, "euhedral_cuda_malloc", MALLOC);
            this.free = bind(linker, symbols, "euhedral_cuda_free", FREE);
            this.deviceMemoryInfo = bind(linker, symbols, "euhedral_cuda_device_memory_info", DEVICE_MEMORY_INFO);
            this.copyHostToDevice = bind(linker, symbols, "euhedral_cuda_copy_host_to_device", COPY);
            this.copyDeviceToHost = bind(linker, symbols, "euhedral_cuda_copy_device_to_host", COPY);
            this.embedQ3 = bind(linker, symbols, "euhedral_cuda_embed_q3", EMBED_Q3);
            this.synchronize = bind(linker, symbols, "euhedral_cuda_synchronize", SYNCHRONIZE);
            this.rmsNormBf16 = bind(linker, symbols, "euhedral_cuda_rms_norm_bf16", RMS_NORM_BF16);
            this.rmsNormUnitOffsetBf16 =
                    bind(linker, symbols, "euhedral_cuda_rms_norm_unit_offset_bf16", RMS_NORM_UNIT_OFFSET_BF16);
            this.linearQ3Bf16 = bind(linker, symbols, "euhedral_cuda_linear_q3_bf16", LINEAR_Q3_BF16);
            this.linearQuantizedBf16 =
                    bind(linker, symbols, "euhedral_cuda_linear_quantized_bf16", LINEAR_QUANTIZED_BF16);
            this.linearBf16ToFloat = bind(linker, symbols, "euhedral_cuda_linear_bf16_to_float", LINEAR_BF16_TO_FLOAT);
            this.gdnControlFp32 = bind(linker, symbols, "euhedral_cuda_gdn_control_fp32", GDN_CONTROL_FP32);
            this.gdnConvolutionBf16 = bind(linker, symbols, "euhedral_cuda_gdn_convolution_bf16", GDN_CONVOLUTION_BF16);
            this.gdnRecurrenceBf16 = bind(linker, symbols, "euhedral_cuda_gdn_recurrence_bf16", GDN_RECURRENCE_BF16);
            this.gdnGatedRmsNormBf16 =
                    bind(linker, symbols, "euhedral_cuda_gdn_gated_rms_norm_bf16", GDN_GATED_RMS_NORM_BF16);
            this.residualAddBf16 = bind(linker, symbols, "euhedral_cuda_residual_add_bf16", RESIDUAL_ADD_BF16);
            this.swiGluBf16 = bind(linker, symbols, "euhedral_cuda_swiglu_bf16", SWIGLU_BF16);
            this.zeroDeviceMemory = bind(linker, symbols, "euhedral_cuda_zero_device_memory", ZERO_DEVICE_MEMORY);
        } catch (RuntimeException exception) {
            loadedLibraryArena.close();
            throw exception;
        }
    }

    @Override
    public long allocate(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        try {
            MemorySegment address = (MemorySegment) malloc.invokeExact(byteSize);
            long value = address.address();
            if (value == 0) throw new GpuMemoryException("CUDA allocation returned a null address");
            return value;
        } catch (GpuMemoryException exception) {
            throw exception;
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA allocation invocation failed", throwable);
        }
    }

    @Override
    public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
        ensureOpen();
        requireTransferSize(source, byteSize, "source");
        requireDeviceAddress(destination);
        int status = invokeCopy(
                copyHostToDevice, MemorySegment.ofAddress(destination), source, byteSize, "host-to-device copy");
        if (status != 0) throw new GpuMemoryException("host-to-device copy", status);
    }

    @Override
    public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        ensureOpen();
        requireTransferSize(destination, byteSize, "destination");
        requireDeviceAddress(source);
        int status = invokeCopy(
                copyDeviceToHost, destination, MemorySegment.ofAddress(source), byteSize, "device-to-host copy");
        if (status != 0) throw new GpuMemoryException("device-to-host copy", status);
    }

    @Override
    public void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize) {
        ensureOpen();
        requireDeviceAddress(tokenIdsAddress);
        requireDeviceAddress(embeddingAddress);
        requireDeviceAddress(hiddenStateAddress);
        if (embeddingByteSize <= 0 || tokenCount <= 0 || vocabularySize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("Q3 embedding sizes must be positive");
        }
        if (hiddenSize % 64 != 0) {
            throw new IllegalArgumentException("Q3 embedding hidden size must be divisible by 64");
        }
        int status;
        try {
            status = (int) embedQ3.invokeExact(
                    MemorySegment.ofAddress(tokenIdsAddress),
                    MemorySegment.ofAddress(embeddingAddress),
                    MemorySegment.ofAddress(hiddenStateAddress),
                    tokenCount,
                    vocabularySize,
                    hiddenSize,
                    embeddingByteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("Q3 embedding invocation failed", throwable);
        }
        if (status != 0) {
            String operation = status == CUDA_FORMAT_MISMATCH ? "Q3 embedding format/layout mismatch" : "Q3 embedding";
            throw new GpuMemoryException(operation, status);
        }
    }

    @Override
    public void synchronize() {
        ensureOpen();
        int status;
        try {
            status = (int) synchronize.invokeExact();
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA device synchronization invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("CUDA device synchronization", status);
    }

    @Override
    public void rmsNormBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        ensureOpen();
        requireAddresses(inputAddress, weightAddress, outputAddress);
        if (rows <= 0 || width <= 0 || !Float.isFinite(epsilon) || epsilon < 0) {
            throw new IllegalArgumentException("RMS norm dimensions and epsilon are invalid");
        }
        int status;
        try {
            status = (int) rmsNormBf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    width,
                    epsilon);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("BF16 RMS norm invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("BF16 RMS norm", status);
    }

    @Override
    public void rmsNormUnitOffsetBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        ensureOpen();
        requireAddresses(inputAddress, weightAddress, outputAddress);
        if (rows <= 0 || width <= 0 || !Float.isFinite(epsilon) || epsilon < 0) {
            throw new IllegalArgumentException("RMS norm dimensions and epsilon are invalid");
        }
        int status;
        try {
            status = (int) rmsNormUnitOffsetBf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    width,
                    epsilon);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("unit-offset BF16 RMS norm invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("unit-offset BF16 RMS norm", status);
    }

    @Override
    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("Q3 linear dimensions and payload size must be positive");
        }
        int status;
        try {
            status = (int) linearQ3Bf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightsAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    inFeatures,
                    outFeatures,
                    weightsByteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("Q3 linear invocation failed", throwable);
        }
        if (status != 0) {
            String operation = status == CUDA_FORMAT_MISMATCH ? "Q3 linear format/layout mismatch" : "Q3 linear";
            throw new GpuMemoryException(operation, status);
        }
    }

    @Override
    public void linearQ4Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        linearQuantizedBf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize, 4);
    }

    @Override
    public void linearQ5Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        linearQuantizedBf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize, 5);
    }

    private void linearQuantizedBf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize,
            int bits) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("quantized linear dimensions and payload size must be positive");
        }
        int status;
        try {
            status = (int) linearQuantizedBf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightsAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    inFeatures,
                    outFeatures,
                    weightsByteSize,
                    bits);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("Q" + bits + " linear invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("Q" + bits + " linear", status);
    }

    @Override
    public void linearBf16ToFloat(
            long inputAddress, long weightsAddress, long outputAddress, int rows, int inFeatures, int outFeatures) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0) {
            throw new IllegalArgumentException("BF16 linear dimensions must be positive");
        }
        invokeLayer(
                "BF16-to-FP32 linear",
                linearBf16ToFloat,
                MemorySegment.ofAddress(inputAddress),
                MemorySegment.ofAddress(weightsAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                inFeatures,
                outFeatures);
    }

    @Override
    public void gdnControlFp32(
            long aProjectionAddress,
            long bProjectionAddress,
            long aLogAddress,
            long dtBiasAddress,
            long gOutputAddress,
            long betaOutputAddress,
            int rows,
            int heads) {
        ensureOpen();
        requireAddresses(
                aProjectionAddress, bProjectionAddress, aLogAddress, dtBiasAddress, gOutputAddress, betaOutputAddress);
        if (rows <= 0 || heads <= 0) throw new IllegalArgumentException("GDN control dimensions must be positive");
        invokeLayer(
                "GDN control",
                gdnControlFp32,
                MemorySegment.ofAddress(aProjectionAddress),
                MemorySegment.ofAddress(bProjectionAddress),
                MemorySegment.ofAddress(aLogAddress),
                MemorySegment.ofAddress(dtBiasAddress),
                MemorySegment.ofAddress(gOutputAddress),
                MemorySegment.ofAddress(betaOutputAddress),
                rows,
                heads);
    }

    @Override
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
        ensureOpen();
        requireAddresses(
                queryKeyAddress, valueZAddress, convolutionWeightsAddress, convolutionStateAddress, outputAddress);
        if (rows <= 0
                || queryKeyWidth <= 0
                || valueWidth <= 0
                || convolutionWidth != queryKeyWidth + valueWidth
                || kernelSize < 2
                || kernelSize > 32) {
            throw new IllegalArgumentException("GDN convolution dimensions are invalid");
        }
        invokeLayer(
                "GDN convolution",
                gdnConvolutionBf16,
                MemorySegment.ofAddress(queryKeyAddress),
                MemorySegment.ofAddress(valueZAddress),
                MemorySegment.ofAddress(convolutionWeightsAddress),
                MemorySegment.ofAddress(convolutionStateAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                queryKeyWidth,
                valueWidth,
                convolutionWidth,
                kernelSize);
    }

    @Override
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
        ensureOpen();
        requireAddresses(convolvedAddress, gAddress, betaAddress, recurrentStateAddress, outputAddress);
        if (rows <= 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyHeadDim != 128
                || valueHeadDim != 128
                || !Float.isFinite(outputScale)
                || outputScale <= 0) {
            throw new IllegalArgumentException("GDN recurrence dimensions and scale are invalid");
        }
        invokeLayer(
                "GDN recurrence",
                gdnRecurrenceBf16,
                MemorySegment.ofAddress(convolvedAddress),
                MemorySegment.ofAddress(gAddress),
                MemorySegment.ofAddress(betaAddress),
                MemorySegment.ofAddress(recurrentStateAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                keyHeads,
                valueHeads,
                keyHeadDim,
                valueHeadDim,
                outputScale);
    }

    @Override
    public void gdnGatedRmsNormBf16(
            long recurrentAddress,
            long valueZAddress,
            long normWeightAddress,
            long outputAddress,
            int rows,
            int valueHeads,
            int headDim,
            float epsilon) {
        ensureOpen();
        requireAddresses(recurrentAddress, valueZAddress, normWeightAddress, outputAddress);
        if (rows <= 0 || valueHeads <= 0 || headDim != 128 || !Float.isFinite(epsilon) || epsilon < 0) {
            throw new IllegalArgumentException("GDN gated RMSNorm dimensions and epsilon are invalid");
        }
        invokeLayer(
                "GDN gated RMSNorm",
                gdnGatedRmsNormBf16,
                MemorySegment.ofAddress(recurrentAddress),
                MemorySegment.ofAddress(valueZAddress),
                MemorySegment.ofAddress(normWeightAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                valueHeads,
                headDim,
                epsilon);
    }

    @Override
    public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {
        ensureOpen();
        requireAddresses(residualAddress, deltaAddress, outputAddress);
        if (rows <= 0 || width <= 0) throw new IllegalArgumentException("residual dimensions must be positive");
        invokeLayer(
                "BF16 residual add",
                residualAddBf16,
                MemorySegment.ofAddress(residualAddress),
                MemorySegment.ofAddress(deltaAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                width);
    }

    @Override
    public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {
        ensureOpen();
        requireAddresses(gateUpAddress, outputAddress);
        if (rows <= 0 || intermediateSize <= 0)
            throw new IllegalArgumentException("SwiGLU dimensions must be positive");
        invokeLayer(
                "BF16 SwiGLU",
                swiGluBf16,
                MemorySegment.ofAddress(gateUpAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                intermediateSize);
    }

    @Override
    public void zeroDeviceMemory(long address, long byteSize) {
        ensureOpen();
        requireDeviceAddress(address);
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        invokeLayer("CUDA device memory zero", zeroDeviceMemory, MemorySegment.ofAddress(address), byteSize);
    }

    @Override
    public void free(long address) {
        ensureOpen();
        if (address == 0) return;
        int status;
        try {
            status = (int) free.invokeExact(MemorySegment.ofAddress(address));
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA free invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("CUDA free", status);
    }

    /// Returns the current CUDA device's free and total memory, in bytes.
    public DeviceMemoryInfo deviceMemoryInfo() {
        ensureOpen();
        try (Arena queryArena = Arena.ofConfined()) {
            MemorySegment freeBytes = queryArena.allocate(Long.BYTES, Long.BYTES);
            MemorySegment totalBytes = queryArena.allocate(Long.BYTES, Long.BYTES);
            int status;
            try {
                status = (int) deviceMemoryInfo.invokeExact(freeBytes, totalBytes);
            } catch (Throwable throwable) {
                throw new GpuMemoryException("CUDA device memory query invocation failed", throwable);
            }
            if (status != 0) throw new GpuMemoryException("CUDA device memory query", status);
            long free = freeBytes.get(ValueLayout.JAVA_LONG, 0);
            long total = totalBytes.get(ValueLayout.JAVA_LONG, 0);
            if (free < 0 || total <= 0 || free > total) {
                throw new GpuMemoryException("CUDA device memory query returned invalid byte counts");
            }
            return new DeviceMemoryInfo(free, total);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            arena.close();
        }
    }

    private static int invokeCopy(
            MethodHandle handle,
            MemorySegment firstAddress,
            MemorySegment secondAddress,
            long byteSize,
            String operation) {
        try {
            return (int) handle.invokeExact(firstAddress, secondAddress, byteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException(operation + " invocation failed", throwable);
        }
    }

    private void invokeLayer(String operation, MethodHandle handle, Object... arguments) {
        int status;
        try {
            status = (int) handle.invokeWithArguments(arguments);
        } catch (Throwable throwable) {
            throw new GpuMemoryException(operation + " invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException(operation, status);
    }

    private static void requireAddresses(long... addresses) {
        for (long address : addresses) requireDeviceAddress(address);
    }

    private static void requireDeviceAddress(long address) {
        if (address == 0) throw new IllegalArgumentException("device address must be non-null");
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("CUDA memory binding is closed");
    }

    private static void requireTransferSize(MemorySegment hostSegment, long byteSize, String name) {
        Objects.requireNonNull(hostSegment, name);
        if (byteSize < 0 || byteSize > hostSegment.byteSize()) {
            throw new IllegalArgumentException(name + " does not contain byteSize bytes");
        }
    }

    /// A snapshot of device memory capacity, in bytes.
    public record DeviceMemoryInfo(long freeBytes, long totalBytes) {}
}
