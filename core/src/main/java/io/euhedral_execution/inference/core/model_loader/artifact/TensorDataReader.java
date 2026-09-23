package io.euhedral_execution.inference.core.model_loader.artifact;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Reads an unmodified tensor payload from a Qwen artifact file.
public final class TensorDataReader {

    private TensorDataReader() {}

    public static MemorySegment read(Path path, TensorDescriptor tensor, Arena arena) throws IOException {
        if (path == null) {
            throw new QwenArtifactFormatException("input path is null");
        }
        if (tensor == null) {
            throw new QwenArtifactFormatException("tensor descriptor is null");
        }
        if (arena == null) {
            throw new QwenArtifactFormatException("arena is null");
        }
        long payloadEnd = QwenArtifactCodec.checkedEnd(tensor.dataOffset(), tensor.byteSize(), "tensor data");

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (payloadEnd > channel.size()) {
                throw new QwenArtifactFormatException("tensor payload extends beyond the file: " + tensor.name());
            }
            MemorySegment payload = arena.allocate(tensor.byteSize());
            ArtifactFileAccess.readFully(channel, tensor.dataOffset(), payload, "tensor payload");
            return payload;
        }
    }
}
