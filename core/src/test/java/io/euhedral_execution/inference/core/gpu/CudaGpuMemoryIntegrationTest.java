package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class CudaGpuMemoryIntegrationTest {

    @Test
    void copiesDeterministicBytesThroughCudaMemory() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(libraryPath), "native CUDA library is missing: " + libraryPath);

        byte[] expected = new byte[64];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = (byte) (i * 37 + 11);
        }

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(expected.length, 1);
            MemorySegment destination = arena.allocate(expected.length, 1);
            source.asByteBuffer().put(expected);
            long device = gpu.allocate(expected.length);
            try {
                gpu.copyHostToDevice(device, source, expected.length);
                gpu.copyDeviceToHost(destination, device, expected.length);

                ByteBuffer result = destination.asByteBuffer();
                byte[] actual = new byte[expected.length];
                result.get(actual);
                assertArrayEquals(expected, actual);
            } finally {
                gpu.free(device);
            }
        }
    }

    @Test
    void validatesJavaArguments() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(8, 1);

            assertThrows(IllegalArgumentException.class, () -> gpu.allocate(0));
            assertThrows(IllegalArgumentException.class, () -> gpu.copyHostToDevice(0, host, 1));
            assertThrows(IllegalArgumentException.class, () -> gpu.copyDeviceToHost(host, 0, 1));
            assertThrows(IllegalArgumentException.class, () -> gpu.copyHostToDevice(1, host, 9));
        }
    }

    @Test
    void rejectsUseAfterClose() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);

        gpu.close();

        assertThrows(IllegalStateException.class, () -> gpu.allocate(1));
        assertThrows(IllegalStateException.class, () -> gpu.free(0));
        gpu.close();
    }
}
