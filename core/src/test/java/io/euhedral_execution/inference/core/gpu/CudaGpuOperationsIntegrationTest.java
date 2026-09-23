package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class CudaGpuOperationsIntegrationTest {

    @Test
    void q3LinearReferenceDecodesSubnormalScale() {
        int width = 128;
        byte[] weights = q3Weights(1, width);
        int scaleOffset = (2 * 24 + 255) & ~255;
        ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN).putShort(scaleOffset, (short) 1);
        short[] input = new short[width];
        input[0] = floatToBf16(1.0f);

        assertEquals(
                floatToBf16(-4.0f * Float.float16ToFloat((short) 1)), linearReference(input, weights, 1, width, 1)[0]);
    }

    @Test
    void bf16RmsNormMatchesCpuReference() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(libraryPath));
        int rows = 2;
        int width = 128;
        float epsilon = 1.0e-5f;
        short[] input = new short[rows * width];
        short[] weight = new short[width];
        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 19 - 9) * 0.125f);
        for (int i = 0; i < weight.length; i++) weight[i] = floatToBf16(0.75f + (i % 7) * 0.03125f);
        short[] expected = rmsNorm(input, weight, rows, width, epsilon);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            long inputDevice = upload(gpu, arena, input);
            long weightDevice = upload(gpu, arena, weight);
            long outputDevice = gpu.allocate((long) expected.length * 2);
            try {
                gpu.rmsNormBf16(inputDevice, weightDevice, outputDevice, rows, width, epsilon);
                short[] actual = download(gpu, arena, outputDevice, expected.length);
                assertBf16Equals(expected, actual, 0.002f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void firstRmsNormOnAnotherWorkerInitializesCudaContext() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        int width = 128;
        short[] input = new short[width];
        short[] weight = new short[width];
        for (int col = 0; col < width; col++) {
            input[col] = floatToBf16((col % 13 - 6) * 0.125f);
            weight[col] = floatToBf16(0.75f);
        }
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined();
                var worker = Executors.newSingleThreadExecutor()) {
            long inputDevice = upload(gpu, arena, input);
            long weightDevice = upload(gpu, arena, weight);
            long outputDevice = gpu.allocate((long) width * Short.BYTES);
            try {
                worker.submit(() -> gpu.rmsNormBf16(inputDevice, weightDevice, outputDevice, 1, width, 1.0e-5f))
                        .get(10, TimeUnit.SECONDS);
                assertBf16Equals(
                        rmsNorm(input, weight, 1, width, 1.0e-5f), download(gpu, arena, outputDevice, width), 0.002f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void q3RowSplitLinearMatchesCpuReference() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        int rows = 2;
        int inFeatures = 128;
        int outFeatures = 3;
        short[] input = new short[rows * inFeatures];
        for (int i = 0; i < input.length; i++) input[i] = floatToBf16((i % 13 - 6) * 0.2f);
        byte[] weights = q3Weights(outFeatures, inFeatures);
        short[] expected = linearReference(input, weights, rows, inFeatures, outFeatures);

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            long inputDevice = upload(gpu, arena, input);
            long weightsDevice = upload(gpu, arena, weights);
            long outputDevice = gpu.allocate((long) expected.length * 2);
            try {
                gpu.linearQ3Bf16(
                        inputDevice, weightsDevice, outputDevice, rows, inFeatures, outFeatures, weights.length);
                short[] actual = download(gpu, arena, outputDevice, expected.length);
                assertBf16Equals(expected, actual, 0.02f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightsDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void q3LinearHandlesZeroAndSubnormalScalesOnGpu() {
        int width = 128;
        byte[] weights = q3Weights(2, width);
        int scaleOffset = (2 * 2 * 24 + 255) & ~255;
        ByteBuffer scales = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        scales.putShort(scaleOffset, (short) 0);
        scales.putShort(scaleOffset + 2 * Short.BYTES, (short) 1);
        short[] input = new short[width];
        input[0] = floatToBf16(1.0f);
        short[] expected = linearReference(input, weights, 1, width, 2);

        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined()) {
            long inputDevice = upload(gpu, arena, input);
            long weightsDevice = upload(gpu, arena, weights);
            long outputDevice = gpu.allocate((long) expected.length * Short.BYTES);
            try {
                gpu.linearQ3Bf16(inputDevice, weightsDevice, outputDevice, 1, width, 2, weights.length);
                assertBf16Equals(expected, download(gpu, arena, outputDevice, expected.length), 1.0e-8f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightsDevice);
                gpu.free(inputDevice);
            }
        }
    }

    @Test
    void paddedQ3LinearOnAnotherWorkerMatchesCpuReference() throws Exception {
        int width = 192;
        int outputs = 5;
        short[] input = new short[width];
        for (int col = 0; col < width; col++) input[col] = floatToBf16((col % 17 - 8) * 0.125f);
        byte[] weights = q3Weights(outputs, width);
        try (CudaGpuMemory gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
                Arena arena = Arena.ofConfined();
                var worker = Executors.newSingleThreadExecutor()) {
            long inputDevice = upload(gpu, arena, input);
            long weightDevice = upload(gpu, arena, weights);
            long outputDevice = gpu.allocate((long) outputs * Short.BYTES);
            try {
                worker.submit(() -> gpu.linearQ3Bf16(
                                inputDevice, weightDevice, outputDevice, 1, width, outputs, weights.length))
                        .get(10, TimeUnit.SECONDS);
                assertBf16Equals(
                        linearReference(input, weights, 1, width, outputs),
                        download(gpu, arena, outputDevice, outputs),
                        0.03f);
            } finally {
                gpu.free(outputDevice);
                gpu.free(weightDevice);
                gpu.free(inputDevice);
            }
        }
    }

    static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        MemorySegment host = arena.allocate((long) values.length * 2, 2);
        ByteBuffer buffer = host.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        for (short value : values) buffer.putShort(value);
        long device = gpu.allocate(host.byteSize());
        gpu.copyHostToDevice(device, host, host.byteSize());
        return device;
    }

    static long upload(CudaGpuMemory gpu, Arena arena, byte[] values) {
        MemorySegment host = arena.allocate(values.length, 1);
        host.asByteBuffer().put(values);
        long device = gpu.allocate(values.length);
        gpu.copyHostToDevice(device, host, values.length);
        return device;
    }

    static short[] download(CudaGpuMemory gpu, Arena arena, long device, int count) {
        MemorySegment host = arena.allocate((long) count * 2, 2);
        gpu.copyDeviceToHost(host, device, host.byteSize());
        ByteBuffer buffer = host.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        short[] result = new short[count];
        for (int i = 0; i < count; i++) result[i] = buffer.getShort();
        return result;
    }

    static void assertBf16Equals(short[] expected, short[] actual, float tolerance) {
        for (int i = 0; i < expected.length; i++) {
            float difference = Math.abs(bf16ToFloat(expected[i]) - bf16ToFloat(actual[i]));
            assertTrue(difference <= tolerance, "index " + i + ": " + difference);
        }
    }

    static short[] rmsNorm(short[] input, short[] weight, int rows, int width, float epsilon) {
        short[] result = new short[input.length];
        for (int row = 0; row < rows; row++) {
            float sum = 0;
            for (int col = 0; col < width; col++) {
                float value = bf16ToFloat(input[row * width + col]);
                sum += value * value;
            }
            float inverse = (float) (1.0 / Math.sqrt(sum / width + epsilon));
            for (int col = 0; col < width; col++) {
                result[row * width + col] =
                        floatToBf16(bf16ToFloat(input[row * width + col]) * inverse * bf16ToFloat(weight[col]));
            }
        }
        return result;
    }

    static byte[] q3Weights(int rows, int width) {
        int groups = ((width + 127) / 128) * 2;
        int base = rows * groups * 24;
        int scaleOffset = (base + 255) & ~255;
        byte[] payload = new byte[scaleOffset + rows * groups * 2];
        ByteBuffer scales = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groups; group++) {
                int offset = scaleOffset + (row * groups + group) * 2;
                scales.putShort(offset, floatToFp16(0.125f * (group + 1)));
                for (int k = 0; k < 64; k++) {
                    int code = ((row * 11 + group * 5 + k) % 8) - 4;
                    int bit = k * 3;
                    int byteOffset = row * groups * 24 + group * 24 + (bit >>> 3);
                    int shift = bit & 7;
                    int packed = (code & 7) << shift;
                    payload[byteOffset] |= (byte) packed;
                    if (shift > 5) payload[byteOffset + 1] |= (byte) (packed >>> 8);
                }
            }
        }
        return payload;
    }

    static short[] linearReference(short[] input, byte[] weights, int rows, int width, int outputs) {
        int groups = ((width + 127) / 128) * 2;
        int scaleOffset = (outputs * groups * 24 + 255) & ~255;
        short[] result = new short[rows * outputs];
        ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++)
            for (int out = 0; out < outputs; out++) {
                float sum = 0;
                for (int k = 0; k < width; k++) {
                    int group = k / 64;
                    int bit = (k % 64) * 3;
                    int offset = out * groups * 24 + group * 24 + (bit >>> 3);
                    int packed = (weights[offset] & 255) | ((weights[offset + 1] & 255) << 8);
                    int code = (packed >>> (bit & 7)) & 7;
                    if (code >= 4) code -= 8;
                    float scale = Float.float16ToFloat(buffer.getShort(scaleOffset + (out * groups + group) * 2));
                    sum += bf16ToFloat(input[row * width + k]) * code * scale;
                }
                result[row * outputs + out] = floatToBf16(sum);
            }
        return result;
    }

    static float bf16ToFloat(short value) {
        return Float.intBitsToFloat((value & 0xffff) << 16);
    }

    static short floatToBf16(float value) {
        int bits = Float.floatToRawIntBits(value);
        return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
    }

    private static short floatToFp16(float value) {
        int bits = Float.floatToRawIntBits(value), sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 255) - 112, mantissa = (bits >>> 13) & 1023;
        return (short) (sign | Math.max(0, Math.min(31, exponent)) << 10 | mantissa);
    }
}
