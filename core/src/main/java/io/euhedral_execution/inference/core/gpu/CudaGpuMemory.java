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
public final class CudaGpuMemory implements GpuMemory, AutoCloseable {

    private static final FunctionDescriptor MALLOC = FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    private static final FunctionDescriptor FREE = FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);
    private static final FunctionDescriptor COPY = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);

    private final Arena arena;
    private final MethodHandle malloc;
    private final MethodHandle free;
    private final MethodHandle copyHostToDevice;
    private final MethodHandle copyDeviceToHost;
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
            this.copyHostToDevice = bind(linker, symbols, "euhedral_cuda_copy_host_to_device", COPY);
            this.copyDeviceToHost = bind(linker, symbols, "euhedral_cuda_copy_device_to_host", COPY);
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
    public void free(long address) {
        ensureOpen();
        if (address == 0) {
            return;
        }
        try {
            free.invokeExact(MemorySegment.ofAddress(address));
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA free invocation failed", throwable);
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
}
