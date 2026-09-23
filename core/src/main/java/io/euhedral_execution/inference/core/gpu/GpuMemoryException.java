package io.euhedral_execution.inference.core.gpu;

/// Reports a native CUDA or Java FFM failure.
public final class GpuMemoryException extends RuntimeException {

    private final int status;

    public GpuMemoryException(String message) {
        this(message, Integer.MIN_VALUE, null);
    }

    public GpuMemoryException(String message, Throwable cause) {
        this(message, Integer.MIN_VALUE, cause);
    }

    public GpuMemoryException(String operation, int status) {
        this(operation + " failed with native status " + status, status, null);
    }

    public int status() {
        return status;
    }

    private GpuMemoryException(String message, int status, Throwable cause) {
        super(message, cause);
        this.status = status;
    }
}
