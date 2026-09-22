package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Writes the metadata and tensor table of a version 1 Qwen artifact.
///
/// The no-payload overload is useful for metadata-only artifacts and accepts only zero-sized
/// tensor descriptors. The payload overload writes each byte array at the descriptor's absolute
/// `dataOffset`; its array order must match the tensor table order.
public final class QwenArtifactWriter {

    public static final int HEADER_SIZE = QwenArtifactHeader.BYTE_SIZE;

    private QwenArtifactWriter() {}

    public static long metadataSize(QwenConfig config) {
        try {
            return QwenArtifactCodec.encodeMetadata(config).length;
        } catch (QwenArtifactFormatException exception) {
            throw new IllegalArgumentException(exception.getMessage(), exception);
        }
    }

    public static long tensorTableSize(TensorDescriptor[] tensors) {
        try {
            return QwenArtifactCodec.encodeTensorTable(tensors).length;
        } catch (QwenArtifactFormatException exception) {
            throw new IllegalArgumentException(exception.getMessage(), exception);
        }
    }

    public static void write(Path path, QwenArtifact artifact) throws IOException {
        write(path, artifact, null);
    }

    public static void write(Path path, QwenArtifact artifact, byte[][] tensorData) throws IOException {
        if (path == null) {
            throw new QwenArtifactFormatException("output path is null");
        }
        if (artifact == null) {
            throw new QwenArtifactFormatException("artifact is null");
        }
        if (artifact.header() == null) {
            throw new QwenArtifactFormatException("artifact header is null");
        }

        byte[] metadata = QwenArtifactCodec.encodeMetadata(artifact.config());
        byte[] tensorTable = QwenArtifactCodec.encodeTensorTable(artifact.tensors());
        validateHeader(artifact.header(), artifact.tensors(), metadata, tensorTable);
        long fileSize = validateTensorData(artifact.header(), artifact.tensors(), tensorData);

        ByteBuffer header = ByteBuffer.allocate(HEADER_SIZE).order(QwenArtifactCodec.BYTE_ORDER);
        header.putInt(artifact.header().magic());
        header.putInt(artifact.header().version());
        header.putLong(artifact.header().metadataOffset());
        header.putLong(artifact.header().metadataSize());
        header.putLong(artifact.header().tensorTableOffset());
        header.putInt(artifact.header().tensorCount());
        header.putInt(0);
        header.putLong(artifact.header().tensorDataOffset());
        header.flip();

        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            extendFile(channel, fileSize);
            writeAt(channel, 0, header);
            writeAt(channel, artifact.header().metadataOffset(), ByteBuffer.wrap(metadata));
            writeAt(channel, artifact.header().tensorTableOffset(), ByteBuffer.wrap(tensorTable));
            if (tensorData != null) {
                for (int i = 0; i < tensorData.length; i++) {
                    if (tensorData[i].length > 0) {
                        writeAt(channel, artifact.tensors()[i].dataOffset(), ByteBuffer.wrap(tensorData[i]));
                    }
                }
            }
        }
    }

    private static void validateHeader(
            QwenArtifactHeader header, TensorDescriptor[] tensors, byte[] metadata, byte[] tensorTable)
            throws QwenArtifactFormatException {
        if (header.magic() != QwenArtifactHeader.MAGIC) {
            throw new QwenArtifactFormatException("unsupported artifact magic");
        }
        if (header.version() != QwenArtifactHeader.VERSION) {
            throw new QwenArtifactFormatException("unsupported artifact version: " + header.version());
        }
        if (header.metadataOffset() != HEADER_SIZE) {
            throw new QwenArtifactFormatException("metadata must immediately follow the header");
        }
        if (header.metadataSize() != metadata.length) {
            throw new QwenArtifactFormatException("header metadata size does not match encoded metadata");
        }
        long metadataEnd = QwenArtifactCodec.checkedEnd(header.metadataOffset(), header.metadataSize(), "metadata");
        if (header.tensorTableOffset() != metadataEnd) {
            throw new QwenArtifactFormatException("tensor table must immediately follow metadata");
        }
        long tensorTableEnd =
                QwenArtifactCodec.checkedEnd(header.tensorTableOffset(), tensorTable.length, "tensor table");
        if (header.tensorDataOffset() != tensorTableEnd) {
            throw new QwenArtifactFormatException("tensor data must immediately follow the tensor table");
        }
        if (header.tensorCount() < 0 || header.tensorCount() > QwenArtifactCodec.MAX_COUNT) {
            throw new QwenArtifactFormatException("tensor count is outside the supported range");
        }
        if (header.tensorCount() != tensors.length) {
            throw new QwenArtifactFormatException("header tensor count does not match tensor table");
        }
    }

    private static long validateTensorData(QwenArtifactHeader header, TensorDescriptor[] tensors, byte[][] tensorData)
            throws QwenArtifactFormatException {
        long fileSize = header.tensorDataOffset();
        if (tensorData != null && tensorData.length != tensors.length) {
            throw new QwenArtifactFormatException("tensor data count does not match tensor table");
        }
        for (int i = 0; i < tensors.length; i++) {
            TensorDescriptor tensor = tensors[i];
            if (tensor.dataOffset() < header.tensorDataOffset()) {
                throw new QwenArtifactFormatException("tensor data offset precedes tensor data section");
            }
            long tensorEnd = QwenArtifactCodec.checkedEnd(tensor.dataOffset(), tensor.byteSize(), "tensor data");
            fileSize = Math.max(fileSize, tensorEnd);
            if (tensorData == null) {
                if (tensor.byteSize() != 0) {
                    throw new QwenArtifactFormatException(
                            "tensor payload is required for non-empty tensor: " + tensor.name());
                }
            } else {
                if (tensorData[i] == null) {
                    throw new QwenArtifactFormatException("tensor payload is null: " + tensor.name());
                }
                if (tensor.byteSize() != tensorData[i].length) {
                    throw new QwenArtifactFormatException(
                            "tensor payload size does not match descriptor: " + tensor.name());
                }
            }
        }
        return fileSize;
    }

    private static void extendFile(FileChannel channel, long fileSize) throws IOException {
        channel.truncate(0);
        if (fileSize > 0) {
            channel.position(fileSize - 1);
            channel.write(ByteBuffer.wrap(new byte[] {0}));
        }
    }

    private static void writeAt(FileChannel channel, long position, ByteBuffer buffer) throws IOException {
        channel.position(position);
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }
}
