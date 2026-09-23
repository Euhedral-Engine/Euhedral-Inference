package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDataReader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;

/// Loads one unmodified tensor payload into GPU memory.
public final class TensorLoader {

    private TensorLoader() {}

    public static TensorHandle load(Path artifactPath, TensorDescriptor descriptor, GpuMemory gpuMemory)
            throws IOException {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(gpuMemory, "gpuMemory");

        long deviceAddress = 0;
        boolean allocated = false;
        try (Arena hostArena = Arena.ofConfined()) {
            MemorySegment payload = TensorDataReader.read(artifactPath, descriptor, hostArena);
            deviceAddress = gpuMemory.allocate(descriptor.byteSize());
            allocated = true;
            gpuMemory.copyHostToDevice(deviceAddress, payload, descriptor.byteSize());
            return new TensorHandle(
                    descriptor.name(),
                    descriptor.shape(),
                    descriptor.dataType(),
                    descriptor.format(),
                    descriptor.layout(),
                    deviceAddress,
                    descriptor.byteSize());
        } catch (Throwable failure) {
            if (allocated) {
                try {
                    gpuMemory.free(deviceAddress);
                } catch (Throwable cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            return propagate(failure);
        }
    }

    private static TensorHandle propagate(Throwable failure) throws IOException {
        if (failure instanceof IOException exception) {
            throw exception;
        }
        if (failure instanceof RuntimeException exception) {
            throw exception;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new AssertionError(failure);
    }
}
