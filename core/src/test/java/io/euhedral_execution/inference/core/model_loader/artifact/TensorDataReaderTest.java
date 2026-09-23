package io.euhedral_execution.inference.core.model_loader.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TensorDataReaderTest {

    @TempDir
    Path tempDirectory;

    @Test
    void readsExactPayloadFromAbsoluteOffset() throws Exception {
        Path path = tempDirectory.resolve("weights.edrl");
        int dataOffset = 7;
        byte[] payload = {(byte) 0x00, (byte) 0x7F, (byte) 0x80, (byte) 0xFF};
        byte[] file = new byte[dataOffset + payload.length];
        System.arraycopy(payload, 0, file, dataOffset, payload.length);
        Files.write(path, file);

        ByteBuffer buffer = TensorDataReader.read(path, descriptor(dataOffset, payload.length));

        assertEquals(0, buffer.position());
        assertEquals(payload.length, buffer.remaining());
        assertArrayEquals(payload, readBytes(buffer));
    }

    @Test
    void rejectsTruncatedPayload() throws Exception {
        Path path = tempDirectory.resolve("truncated.edrl");
        Files.write(path, new byte[10]);

        assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(7, 4)));
    }

    @Test
    void readsZeroLengthTensorAsEmptyReadyBuffer() throws Exception {
        Path path = tempDirectory.resolve("empty.edrl");
        Files.write(path, new byte[8]);

        ByteBuffer buffer = TensorDataReader.read(path, descriptor(8, 0));

        assertEquals(0, buffer.position());
        assertEquals(0, buffer.remaining());
    }

    @Test
    void rejectsNegativeOffsetAndSize() throws Exception {
        Path path = tempDirectory.resolve("invalid-range.edrl");
        Files.write(path, new byte[1]);

        assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(-1, 0)));
        assertThrows(QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(0, -1)));
    }

    @Test
    void rejectsOverflowingRange() throws Exception {
        Path path = tempDirectory.resolve("overflowing-range.edrl");
        Files.write(path, new byte[1]);

        assertThrows(
                QwenArtifactFormatException.class, () -> TensorDataReader.read(path, descriptor(Long.MAX_VALUE, 1)));
    }

    @Test
    void rejectsSizeTooLargeForByteBuffer() throws Exception {
        Path path = tempDirectory.resolve("oversized.edrl");
        Files.write(path, new byte[1]);

        assertThrows(
                QwenArtifactFormatException.class,
                () -> TensorDataReader.read(path, descriptor(0, (long) Integer.MAX_VALUE + 1)));
    }

    private static TensorDescriptor descriptor(long dataOffset, long byteSize) {
        return new TensorDescriptor(
                "test.tensor", new long[] {4}, TensorDataType.UINT8, WeightFormat.Q4, dataOffset, byteSize);
    }

    private static byte[] readBytes(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return bytes;
    }
}
