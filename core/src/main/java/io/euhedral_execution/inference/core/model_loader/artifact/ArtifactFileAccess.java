package io.euhedral_execution.inference.core.model_loader.artifact;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

final class ArtifactFileAccess {

    private ArtifactFileAccess() {}

    static void readFully(FileChannel channel, long offset, ByteBuffer destination, String field) throws IOException {
        long position = offset;
        while (destination.hasRemaining()) {
            int read = channel.read(destination, position);
            if (read < 0) {
                throw new QwenArtifactFormatException("artifact is truncated while reading " + field);
            }
            if (read == 0) {
                throw new QwenArtifactFormatException("unable to read " + field);
            }
            position += read;
        }
    }
}
