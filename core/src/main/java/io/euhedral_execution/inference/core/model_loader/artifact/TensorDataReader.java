package io.euhedral_execution.inference.core.model_loader.artifact;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Reads an unmodified tensor payload from a Qwen artifact file.
public final class TensorDataReader {

    private TensorDataReader() {}

    public static ByteBuffer read(Path path, TensorDescriptor tensor) throws IOException {
        if (path == null) {
            throw new QwenArtifactFormatException("input path is null");
        }
        if (tensor == null) {
            throw new QwenArtifactFormatException("tensor descriptor is null");
        }
        if (tensor.byteSize() > Integer.MAX_VALUE) {
            throw new QwenArtifactFormatException("tensor byte size is too large for a ByteBuffer");
        }
        long payloadEnd = QwenArtifactCodec.checkedEnd(tensor.dataOffset(), tensor.byteSize(), "tensor data");

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (payloadEnd > channel.size()) {
                throw new QwenArtifactFormatException("tensor payload extends beyond the file: " + tensor.name());
            }
            ByteBuffer payload = ByteBuffer.allocate((int) tensor.byteSize());
            ArtifactFileAccess.readFully(channel, tensor.dataOffset(), payload, "tensor payload");
            payload.flip();
            return payload;
        }
    }
}
