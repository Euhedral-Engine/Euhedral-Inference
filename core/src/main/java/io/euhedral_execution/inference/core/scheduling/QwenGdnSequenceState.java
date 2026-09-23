package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import java.util.Objects;

/// Persistent GDN convolution and recurrent buffers owned by one sequence.
public final class QwenGdnSequenceState implements AutoCloseable {
    private final QwenExecutionGpu gpu;
    private long convolutionStateAddress;
    private long recurrentStateAddress;
    private boolean closing;
    private boolean closed;

    private QwenGdnSequenceState(QwenExecutionGpu gpu, long convolutionStateAddress, long recurrentStateAddress) {
        this.gpu = gpu;
        this.convolutionStateAddress = convolutionStateAddress;
        this.recurrentStateAddress = recurrentStateAddress;
    }

    /// Allocates zero-initialized GDN state for the supplied layer geometry.
    public static QwenGdnSequenceState allocate(
            QwenExecutionGpu gpu,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            int convolutionKernelDim) {
        Objects.requireNonNull(gpu, "gpu");
        if (keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyHeadDim <= 0
                || valueHeadDim <= 0
                || convolutionKernelDim < 2) {
            throw new IllegalArgumentException("invalid GDN sequence-state dimensions");
        }

        int convolutionChannels = Math.addExact(
                Math.multiplyExact(2, Math.multiplyExact(keyHeads, keyHeadDim)),
                Math.multiplyExact(valueHeads, valueHeadDim));
        long convolutionBytes = Math.multiplyExact(
                Math.multiplyExact((long) convolutionChannels, convolutionKernelDim - 1L), Short.BYTES);
        long recurrentBytes = Math.multiplyExact(
                Math.multiplyExact(Math.multiplyExact((long) valueHeads, keyHeadDim), valueHeadDim), Float.BYTES);

        long convolutionAddress = 0;
        long recurrentAddress = 0;
        try {
            convolutionAddress = allocateRequired(gpu, convolutionBytes);
            recurrentAddress = allocateRequired(gpu, recurrentBytes);
            gpu.zeroDeviceMemory(convolutionAddress, convolutionBytes);
            gpu.zeroDeviceMemory(recurrentAddress, recurrentBytes);
            return new QwenGdnSequenceState(gpu, convolutionAddress, recurrentAddress);
        } catch (Throwable failure) {
            freeAfterFailure(gpu, recurrentAddress, failure);
            freeAfterFailure(gpu, convolutionAddress, failure);
            throw failure;
        }
    }

    public long convolutionStateAddress() {
        ensureOpen();
        return this.convolutionStateAddress;
    }

    public long recurrentStateAddress() {
        ensureOpen();
        return this.recurrentStateAddress;
    }

    @Override
    public void close() {
        if (this.closed) return;
        this.closing = true;
        Throwable failure = null;
        failure = free(this.convolutionStateAddress, failure, true);
        failure = free(this.recurrentStateAddress, failure, false);
        if (failure != null) throw propagate(failure);
        this.closed = true;
    }

    private static long allocateRequired(QwenExecutionGpu gpu, long byteSize) {
        long address = gpu.allocate(byteSize);
        if (address <= 0) throw new IllegalStateException("GPU returned an invalid GDN state address");
        return address;
    }

    private static void freeAfterFailure(QwenExecutionGpu gpu, long address, Throwable failure) {
        if (address == 0) return;
        try {
            gpu.free(address);
        } catch (Throwable cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    private Throwable free(long address, Throwable priorFailure, boolean convolution) {
        if (address == 0) return priorFailure;
        try {
            this.gpu.free(address);
            if (convolution) this.convolutionStateAddress = 0;
            else this.recurrentStateAddress = 0;
            return priorFailure;
        } catch (Throwable failure) {
            if (priorFailure != null) priorFailure.addSuppressed(failure);
            else priorFailure = failure;
            return priorFailure;
        }
    }

    private void ensureOpen() {
        if (this.closing || this.closed) throw new IllegalStateException("GDN sequence state is closed");
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) return runtimeException;
        if (failure instanceof Error error) throw error;
        return new IllegalStateException("failed to release GDN sequence state", failure);
    }
}
