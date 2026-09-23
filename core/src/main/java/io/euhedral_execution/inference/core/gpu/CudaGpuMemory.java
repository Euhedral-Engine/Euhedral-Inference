package io.euhedral_execution.inference.core.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
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
public final class CudaGpuMemory implements QwenExecutionGpu, AutoCloseable {

    private static final FunctionDescriptor MALLOC = FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    private static final FunctionDescriptor FREE = FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
    private static final FunctionDescriptor DEVICE_MEMORY_INFO =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
    private static final FunctionDescriptor COPY = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    private static final FunctionDescriptor EMBED_Q3 = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_INT,
            ValueLayout.JAVA_LONG);
    private static final FunctionDescriptor SYNCHRONIZE = FunctionDescriptor.of(ValueLayout.JAVA_INT);
    private static final int CUDA_FORMAT_MISMATCH = -3;

    private final Arena arena;
    private final MethodHandle malloc;
    private final MethodHandle free;
    private final MethodHandle deviceMemoryInfo;
    private final MethodHandle copyHostToDevice;
    private final MethodHandle copyDeviceToHost;
    private final MethodHandle embedQ3;
    private final MethodHandle synchronize;
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
        } catch (RuntimeException exception) {
            loadedLibraryArena.close();
            throw exception;
        }
    }

    @Override
    public long allocate(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) {
            throw new IllegalArgumentException("byteSize must be positive");
        }
        try {
            MemorySegment address = (MemorySegment) malloc.invokeExact(byteSize);
            long value = address.address();
            if (value == 0) {
                throw new GpuMemoryException("CUDA allocation returned a null address");
            }
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
        if (status != 0) {
            throw new GpuMemoryException("host-to-device copy", status);
        }
    }

    @Override
    public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        ensureOpen();
        requireTransferSize(destination, byteSize, "destination");
        requireDeviceAddress(source);
        int status = invokeCopy(
                copyDeviceToHost, destination, MemorySegment.ofAddress(source), byteSize, "device-to-host copy");
        if (status != 0) {
            throw new GpuMemoryException("device-to-host copy", status);
        }
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
        if (status != 0) {
            throw new GpuMemoryException("CUDA device synchronization", status);
        }
    }

    @Override
    public void free(long address) {
        ensureOpen();
        if (address == 0) {
            return;
        }
        int status;
        try {
            status = (int) free.invokeExact(MemorySegment.ofAddress(address));
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA free invocation failed", throwable);
        }
        if (status != 0) {
            throw new GpuMemoryException("CUDA free", status);
        }
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
            if (status != 0) {
                throw new GpuMemoryException("CUDA device memory query", status);
            }
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

    private static MethodHandle bind(Linker linker, SymbolLookup symbols, String name, FunctionDescriptor descriptor) {
        MemorySegment symbol =
                symbols.find(name).orElseThrow(() -> new GpuMemoryException("native symbol not found: " + name));
        return linker.downcallHandle(symbol, descriptor);
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

    private static void requireDeviceAddress(long address) {
        if (address == 0) {
            throw new IllegalArgumentException("device address must be non-null");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("CUDA memory binding is closed");
        }
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
