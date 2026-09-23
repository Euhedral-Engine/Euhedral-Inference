package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.MemorySegment;

/// The minimal CPU-facing contract for GPU memory ownership and copies.
public interface GpuMemory {

    long allocate(long byteSize);

    void copyHostToDevice(long destination, MemorySegment source, long byteSize);

    void copyDeviceToHost(MemorySegment destination, long source, long byteSize);

    void free(long address);
}
