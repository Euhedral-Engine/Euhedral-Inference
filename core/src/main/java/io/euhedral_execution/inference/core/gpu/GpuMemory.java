package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.MemorySegment;

/// The minimal CPU-facing contract for GPU memory ownership and copies.
public interface GpuMemory {

    long allocate(long byteSize);

    void copyHostToDevice(long destination, MemorySegment source, long byteSize);

    void copyDeviceToHost(MemorySegment destination, long source, long byteSize);

    default void copyDeviceToDevice(long destination, long source, long byteSize) {
        throw new UnsupportedOperationException("device-to-device copy is not implemented by this GPU memory provider");
    }

    void free(long address);
}
