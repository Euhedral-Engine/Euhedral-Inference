package io.euhedral_execution.inference.core.model_loader.artifact;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
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

    static void readFully(FileChannel channel, long offset, MemorySegment destination, String field)
            throws IOException {
        if (offset < 0) {
            throw new QwenArtifactFormatException(field + " offset is negative");
        }
        long position = offset;
        long destinationOffset = 0;
        while (destinationOffset < destination.byteSize()) {
            long chunkSize = Math.min(destination.byteSize() - destinationOffset, Integer.MAX_VALUE);
            ByteBuffer chunk = destination.asSlice(destinationOffset, chunkSize).asByteBuffer();
            readFully(channel, position, chunk, field);
            position += chunkSize;
            destinationOffset += chunkSize;
        }
    }
}
