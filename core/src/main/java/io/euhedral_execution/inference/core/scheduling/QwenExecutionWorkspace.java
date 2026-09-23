package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.util.Objects;

/// GPU storage scoped to one Qwen submission; hidden states are BF16 values in token-major order.
public final class QwenExecutionWorkspace implements AutoCloseable {

    private final GpuMemory gpuMemory;
    private final int tokenCount;
    private final int hiddenSize;
    private final long byteSize;
    private long hiddenStateAddress;
    private boolean closed;

    QwenExecutionWorkspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize) {
        this.gpuMemory = Objects.requireNonNull(gpuMemory, "gpuMemory");
        if (tokenCount <= 0) {
            throw new IllegalArgumentException("tokenCount must be positive");
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive");
        }
        this.tokenCount = tokenCount;
        this.hiddenSize = hiddenSize;
        try {
            this.byteSize = Math.multiplyExact(Math.multiplyExact((long) tokenCount, hiddenSize), Short.BYTES);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("hidden-state workspace size overflows", overflow);
        }
        this.hiddenStateAddress = this.gpuMemory.allocate(this.byteSize);
        if (this.hiddenStateAddress == 0) {
            throw new IllegalStateException("GPU returned a null hidden-state workspace address");
        }
    }

    public int tokenCount() {
        return this.tokenCount;
    }

    public int hiddenSize() {
        return this.hiddenSize;
    }

    public long byteSize() {
        return this.byteSize;
    }

    /// Returns the opaque device address while this submission workspace is live.
    public long hiddenStateAddress() {
        if (this.closed) {
            throw new IllegalStateException("Qwen execution workspace is closed");
        }
        return this.hiddenStateAddress;
    }

    public boolean isClosed() {
        return this.closed;
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        long address = this.hiddenStateAddress;
        this.gpuMemory.free(address);
        this.hiddenStateAddress = 0;
        this.closed = true;
    }
}
