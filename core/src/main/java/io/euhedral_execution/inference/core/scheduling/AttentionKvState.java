package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Sequence-owned, growing BF16 key/value storage for one full-attention layer.
public final class AttentionKvState implements AutoCloseable {

    private final ExecutionGpu gpu;
    private final int keyValueWidth;
    private final List<Long> retiredAddresses = new ArrayList<>();
    private long address;
    private int capacity;
    private int length;
    private boolean closed;

    public AttentionKvState(ExecutionGpu gpu, int keyValueWidth) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        if (keyValueWidth <= 0) throw new IllegalArgumentException("keyValueWidth must be positive");
        this.keyValueWidth = keyValueWidth;
    }

    /// Ensures an append is contiguous and makes room without discarding existing keys or values.
    public void prepareAppend(long startPosition, int tokenCount) {
        ensureOpen();
        if (startPosition != this.length || tokenCount <= 0) {
            throw new IllegalArgumentException("KV append must begin at the current sequence length");
        }
        int required = Math.toIntExact(Math.addExact(startPosition, tokenCount));
        if (required <= this.capacity) return;
        int nextCapacity = this.capacity == 0 ? Math.max(1, required) : this.capacity;
        while (nextCapacity < required) {
            nextCapacity = nextCapacity > Integer.MAX_VALUE / 2 ? required : nextCapacity * 2;
        }
        grow(nextCapacity);
    }

    /// Publishes appended tokens only after the synchronous GPU append completed successfully.
    public void commitAppend(int tokenCount) {
        ensureOpen();
        if (tokenCount <= 0 || (long) this.length + tokenCount > this.capacity) {
            throw new IllegalArgumentException("KV append exceeds reserved capacity");
        }
        this.length = Math.addExact(this.length, tokenCount);
        if (this.gpu.asynchronous()) releaseRetired();
    }

    public long keyCacheAddress() {
        ensureOpen();
        if (this.address == 0) throw new IllegalStateException("KV cache has not been allocated");
        return this.address;
    }

    public long valueCacheAddress() {
        ensureOpen();
        if (this.address == 0) throw new IllegalStateException("KV cache has not been allocated");
        return Math.addExact(
                this.address,
                Math.multiplyExact(Math.multiplyExact((long) this.capacity, this.keyValueWidth), Short.BYTES));
    }

    public int capacity() {
        ensureOpen();
        return this.capacity;
    }

    public int length() {
        ensureOpen();
        return this.length;
    }

    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = null;
        try {
            releaseRetired();
        } catch (Throwable cleanupFailure) {
            failure = combine(failure, cleanupFailure);
        }
        if (this.address != 0) {
            try {
                this.gpu.free(this.address);
                this.address = 0;
                this.capacity = 0;
                this.length = 0;
            } catch (Throwable cleanupFailure) {
                failure = combine(failure, cleanupFailure);
            }
        }
        this.closed = this.address == 0 && this.retiredAddresses.isEmpty();
        if (failure != null) throw propagate(failure);
    }

    private void releaseRetired() {
        Throwable failure = null;
        for (int index = this.retiredAddresses.size() - 1; index >= 0; index--) {
            long retired = this.retiredAddresses.get(index);
            try {
                this.gpu.free(retired);
                this.retiredAddresses.remove(index);
            } catch (Throwable cleanupFailure) {
                failure = combine(failure, cleanupFailure);
            }
        }
        if (failure != null) throw propagate(failure);
    }

    private void grow(int newCapacity) {
        long newByteSize = cacheByteSize(newCapacity);
        long newAddress = this.gpu.allocate(newByteSize);
        if (newAddress == 0) throw new IllegalStateException("GPU returned a null attention KV allocation");
        try {
            if (this.address != 0) {
                long oldPlaneBytes = planeByteSize(this.length);
                if (oldPlaneBytes > 0) {
                    this.gpu.copyDeviceToDevice(newAddress, this.address, oldPlaneBytes);
                    this.gpu.copyDeviceToDevice(
                            newAddress + planeByteSize(newCapacity), valueCacheAddress(), oldPlaneBytes);
                }
            }
        } catch (RuntimeException | Error copyFailure) {
            if (this.gpu.asynchronous()) {
                this.retiredAddresses.add(newAddress);
            } else {
                try {
                    this.gpu.free(newAddress);
                } catch (Throwable cleanupFailure) {
                    copyFailure.addSuppressed(cleanupFailure);
                }
            }
            throw copyFailure;
        }

        long oldAddress = this.address;
        this.address = newAddress;
        this.capacity = newCapacity;
        if (oldAddress != 0) {
            if (this.gpu.asynchronous()) {
                this.retiredAddresses.add(oldAddress);
                return;
            }
            try {
                this.gpu.free(oldAddress);
            } catch (RuntimeException | Error freeFailure) {
                this.retiredAddresses.add(oldAddress);
                throw freeFailure;
            }
        }
    }

    private long cacheByteSize(int tokenCapacity) {
        return Math.multiplyExact(2L, planeByteSize(tokenCapacity));
    }

    private long planeByteSize(int tokenCapacity) {
        return Math.multiplyExact(Math.multiplyExact((long) tokenCapacity, this.keyValueWidth), Short.BYTES);
    }

    private void ensureOpen() {
        if (this.closed) throw new IllegalStateException("attention KV state is closed");
    }

    private static Throwable combine(Throwable first, Throwable next) {
        if (first == null) return next;
        first.addSuppressed(next);
        return first;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) return runtimeException;
        if (failure instanceof Error error) throw error;
        return new IllegalStateException("failed to release attention KV state", failure);
    }
}
