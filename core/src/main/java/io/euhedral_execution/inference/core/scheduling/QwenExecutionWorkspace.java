package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.util.List;
import java.util.Objects;

/// GPU storage scoped to one Qwen submission; hidden states are BF16 values in token-major order.
public final class QwenExecutionWorkspace implements AutoCloseable {

    private final GpuMemory gpuMemory;
    private final int tokenCount;
    private final int hiddenSize;
    private final long byteSize;
    private final long[] projectionAddresses;
    private final long[] projectionByteSizes;
    private long normalizedAddress;
    private long hiddenStateAddress;
    private boolean closed;

    QwenExecutionWorkspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize) {
        this(gpuMemory, tokenCount, hiddenSize, List.of());
    }

    QwenExecutionWorkspace(GpuMemory gpuMemory, int tokenCount, int hiddenSize, List<Integer> projectionWidths) {
        this.gpuMemory = Objects.requireNonNull(gpuMemory, "gpuMemory");
        Objects.requireNonNull(projectionWidths, "projectionWidths");
        if (tokenCount <= 0) {
            throw new IllegalArgumentException("tokenCount must be positive");
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive");
        }
        this.tokenCount = tokenCount;
        this.hiddenSize = hiddenSize;
        this.projectionAddresses = new long[projectionWidths.size()];
        this.projectionByteSizes = new long[projectionWidths.size()];
        try {
            this.byteSize = Math.multiplyExact(Math.multiplyExact((long) tokenCount, hiddenSize), Short.BYTES);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("hidden-state workspace size overflows", overflow);
        }
        for (int index = 0; index < projectionWidths.size(); index++) {
            int width = projectionWidths.get(index);
            if (width <= 0) {
                throw new IllegalArgumentException("projection width must be positive");
            }
            this.projectionByteSizes[index] =
                    Math.multiplyExact(Math.multiplyExact((long) tokenCount, width), Short.BYTES);
        }
    }

    /// Allocate only after the submission owns this object, so partial failure remains reclaimable.
    void allocateBuffers() {
        if (this.closed || this.hiddenStateAddress != 0) {
            throw new IllegalStateException("workspace has already been allocated or closed");
        }
        this.hiddenStateAddress = allocate(this.byteSize);
        if (this.projectionByteSizes.length != 0) {
            this.normalizedAddress = allocate(this.byteSize);
            for (int index = 0; index < this.projectionByteSizes.length; index++) {
                this.projectionAddresses[index] = allocate(this.projectionByteSizes[index]);
            }
        }
    }

    private long allocate(long bytes) {
        long address = this.gpuMemory.allocate(bytes);
        if (address == 0) {
            throw new IllegalStateException("GPU returned a null workspace address");
        }
        return address;
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

    public long normalizedStateAddress() {
        if (this.closed || this.normalizedAddress == 0) {
            throw new IllegalStateException("normalized buffer is unavailable");
        }
        return this.normalizedAddress;
    }

    public long projectionAddress(int index) {
        if (this.closed || this.projectionAddresses[index] == 0) {
            throw new IllegalStateException("projection buffer is unavailable");
        }
        return this.projectionAddresses[index];
    }

    public boolean isClosed() {
        return this.closed;
    }

    @Override
    public void close() {
        if (this.closed) {
            return;
        }
        Throwable failure = null;
        for (int index = 0; index < this.projectionAddresses.length; index++) {
            try {
                if (this.projectionAddresses[index] != 0) {
                    this.gpuMemory.free(this.projectionAddresses[index]);
                    this.projectionAddresses[index] = 0;
                }
            } catch (RuntimeException | Error cleanupFailure) {
                failure = combine(failure, cleanupFailure);
            }
        }
        try {
            if (this.normalizedAddress != 0) {
                this.gpuMemory.free(this.normalizedAddress);
                this.normalizedAddress = 0;
            }
        } catch (RuntimeException | Error cleanupFailure) {
            failure = combine(failure, cleanupFailure);
        }
        try {
            if (this.hiddenStateAddress != 0) {
                this.gpuMemory.free(this.hiddenStateAddress);
                this.hiddenStateAddress = 0;
            }
        } catch (RuntimeException | Error cleanupFailure) {
            failure = combine(failure, cleanupFailure);
        }
        this.closed = this.hiddenStateAddress == 0 && this.normalizedAddress == 0;
        for (long address : this.projectionAddresses) {
            this.closed &= address == 0;
        }
        if (failure != null) {
            if (failure instanceof Error error) {
                throw error;
            }
            throw (RuntimeException) failure;
        }
    }

    private static Throwable combine(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }
}
