package io.euhedral_execution.inference.core.model_loader.artifact;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class QwenArtifactBinaryTest {

    @TempDir
    Path tempDirectory;

    @Test
    void writesAndReadsDeterministicArtifact() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path firstPath = tempDirectory.resolve("first.edrl");
        Path secondPath = tempDirectory.resolve("second.edrl");

        QwenArtifactWriter.write(firstPath, artifact, new byte[][] {{1, 2, 3, 4}, {}});
        QwenArtifactWriter.write(secondPath, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        QwenArtifact read = QwenArtifactReader.read(firstPath);
        assertEquals(artifact.header(), read.header());
        assertConfigEquals(artifact.config(), read.config());
        assertTensorDescriptorsEqual(artifact.tensors(), read.tensors());
        byte[] firstBytes = Files.readAllBytes(firstPath);
        assertArrayEquals(firstBytes, Files.readAllBytes(secondPath));
        assertArrayEquals(
                new byte[] {1, 2, 3, 4},
                java.util.Arrays.copyOfRange(
                        firstBytes, (int) tensors[0].dataOffset(), (int) tensors[0].dataOffset() + 4));
    }

    @Test
    void rejectsInvalidMagic() throws Exception {
        QwenArtifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-magic.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(0, 0);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsTruncatedArtifact() throws Exception {
        QwenArtifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("truncated.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        Files.write(path, java.util.Arrays.copyOf(bytes, bytes.length - 1));

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsTensorDataOutsideFileBounds() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-offset.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        int dataOffsetField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putLong(dataOffsetField, Long.MAX_VALUE);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsTensorCountThatDoesNotMatchTable() throws Exception {
        QwenArtifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-count.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(32, 3);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsUnsupportedVersion() throws Exception {
        QwenArtifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-version.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(4, QwenArtifactHeader.VERSION + 1);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsNonZeroReservedHeaderField() throws Exception {
        QwenArtifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-reserved.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(36, 1);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidTensorDataType() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-data-type.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        int dataTypeField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]) - Long.BYTES;
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putInt(dataTypeField, Integer.MAX_VALUE);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidTensorNameLength() throws Exception {
        QwenArtifact artifact = artifact(sampleConfig(), descriptors(sampleConfig()));
        Path path = tempDirectory.resolve("invalid-name-length.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.BIG_ENDIAN)
                .putInt((int) artifact.header().tensorTableOffset(), -1);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsNegativeTensorByteSize() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-byte-size.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        int byteSizeField = (int) tensorsTableOffset(artifact) + tensorDataOffsetWithinEntry(tensors[0]) + Long.BYTES;
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).putLong(byteSizeField, -1);
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsMalformedMetadataUtf8() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-utf8.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        int activationOffset = (int) artifact.header().metadataOffset()
                + Integer.BYTES * 12
                + Double.BYTES * 3
                + Integer.BYTES
                + Integer.BYTES;
        bytes[activationOffset] = (byte) 0xC3;
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsInvalidMetadataBoolean() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("invalid-boolean.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        int booleanOffset = (int) artifact.header().metadataOffset()
                + Integer.BYTES * 12
                + Double.BYTES * 3
                + Integer.BYTES
                + Integer.BYTES
                + config.hiddenActivation().getBytes(java.nio.charset.StandardCharsets.UTF_8).length
                + Integer.BYTES
                + Integer.BYTES * config.layerTypes().length
                + Integer.BYTES * 4;
        bytes[booleanOffset] = 2;
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsOverlappingTensorPayloads() throws Exception {
        QwenConfig config = sampleConfig();
        QwenArtifact validArtifact = artifact(config, descriptors(config));
        long dataOffset = validArtifact.header().tensorDataOffset();
        TensorDescriptor[] tensors = {
            new TensorDescriptor(
                    "model.embed_tokens.weight",
                    new long[] {32_000, 4_096},
                    TensorDataType.BF16,
                    WeightFormat.BF16,
                    dataOffset,
                    4),
            new TensorDescriptor(
                    "model.norm.weight", new long[] {4_096}, TensorDataType.FP32, WeightFormat.FP32, dataOffset, 4)
        };
        QwenArtifact artifact = new QwenArtifact(validArtifact.header(), config, tensors);
        Path path = tempDirectory.resolve("overlapping-data.edrl");

        assertThrows(
                QwenArtifactFormatException.class,
                () -> QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {5, 6, 7, 8}}));
    }

    @Test
    void rejectsOverlappingTensorPayloadsWhenReading() throws Exception {
        QwenConfig config = sampleConfig();
        TensorDescriptor[] tensors = descriptors(config);
        QwenArtifact artifact = artifact(config, tensors);
        Path path = tempDirectory.resolve("overlapping-data-on-read.edrl");
        QwenArtifactWriter.write(path, artifact, new byte[][] {{1, 2, 3, 4}, {}});

        byte[] bytes = Files.readAllBytes(path);
        int secondDataOffsetField = (int) artifact.header().tensorTableOffset()
                + tensorEntrySize(tensors[0])
                + tensorDataOffsetWithinEntry(tensors[1]);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.BIG_ENDIAN)
                .putLong(secondDataOffsetField, tensors[0].dataOffset())
                .putLong(secondDataOffsetField + Long.BYTES, tensors[0].byteSize());
        Files.write(path, bytes);

        assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
    }

    @Test
    void rejectsMetadataLargerThanReaderLimitBeforeDecoding() throws Exception {
        long metadataSize = QwenArtifactCodec.MAX_METADATA_BYTES + 1;
        long tableOffset = QwenArtifactHeader.BYTE_SIZE + metadataSize;
        Path path = tempDirectory.resolve("oversized-metadata.edrl");
        ByteBuffer header = ByteBuffer.allocate(QwenArtifactHeader.BYTE_SIZE).order(ByteOrder.BIG_ENDIAN);
        header.putInt(QwenArtifactHeader.MAGIC);
        header.putInt(QwenArtifactHeader.VERSION);
        header.putLong(QwenArtifactHeader.BYTE_SIZE);
        header.putLong(metadataSize);
        header.putLong(tableOffset);
        header.putInt(0);
        header.putInt(0);
        header.putLong(tableOffset);
        header.flip();
        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            channel.write(header);
            channel.position(tableOffset - 1);
            channel.write(ByteBuffer.wrap(new byte[] {0}));
        }

        QwenArtifactFormatException exception =
                assertThrows(QwenArtifactFormatException.class, () -> QwenArtifactReader.read(path));
        assertEquals("metadata is too large", exception.getMessage());
    }

    private static QwenConfig sampleConfig() {
        return new QwenConfig(
                32_000,
                4_096,
                2,
                32,
                8,
                128,
                11_008,
                16,
                8,
                64,
                128,
                4,
                1.0e-6,
                1_000_000.0,
                1.0,
                32_768,
                "silu",
                new QwenLayerType[] {QwenLayerType.FULL_ATTENTION, QwenLayerType.GATED_DELTA_NET},
                0,
                0,
                0,
                0,
                true,
                false,
                0);
    }

    private static TensorDescriptor[] descriptors(QwenConfig config) {
        long metadataOffset = QwenArtifactWriter.HEADER_SIZE;
        long metadataSize = QwenArtifactWriter.metadataSize(config);
        long tableOffset = metadataOffset + metadataSize;
        long tableSize = QwenArtifactWriter.tensorTableSize(new TensorDescriptor[] {
            new TensorDescriptor(
                    "model.embed_tokens.weight",
                    new long[] {32_000, 4_096},
                    TensorDataType.BF16,
                    WeightFormat.BF16,
                    0,
                    4),
            new TensorDescriptor("model.norm.weight", new long[] {4_096}, TensorDataType.FP32, WeightFormat.FP32, 0, 0)
        });
        long dataOffset = tableOffset + tableSize;
        return new TensorDescriptor[] {
            new TensorDescriptor(
                    "model.embed_tokens.weight",
                    new long[] {32_000, 4_096},
                    TensorDataType.BF16,
                    WeightFormat.BF16,
                    dataOffset,
                    4),
            new TensorDescriptor(
                    "model.norm.weight", new long[] {4_096}, TensorDataType.FP32, WeightFormat.FP32, dataOffset + 4, 0)
        };
    }

    private static QwenArtifact artifact(QwenConfig config, TensorDescriptor[] tensors) {
        long metadataOffset = QwenArtifactWriter.HEADER_SIZE;
        long metadataSize = QwenArtifactWriter.metadataSize(config);
        long tableOffset = metadataOffset + metadataSize;
        long tensorDataOffset = tableOffset + QwenArtifactWriter.tensorTableSize(tensors);
        return new QwenArtifact(
                new QwenArtifactHeader(
                        QwenArtifactHeader.MAGIC,
                        QwenArtifactHeader.VERSION,
                        metadataOffset,
                        metadataSize,
                        tableOffset,
                        tensors.length,
                        tensorDataOffset),
                config,
                tensors);
    }

    private static long tensorsTableOffset(QwenArtifact artifact) {
        return artifact.header().tensorTableOffset();
    }

    private static int tensorDataOffsetWithinEntry(TensorDescriptor descriptor) {
        int nameLength = descriptor.name().getBytes(java.nio.charset.StandardCharsets.UTF_8).length;
        return Integer.BYTES + nameLength + Integer.BYTES + Long.BYTES * descriptor.shape().length + Integer.BYTES * 2;
    }

    private static int tensorEntrySize(TensorDescriptor descriptor) {
        return tensorDataOffsetWithinEntry(descriptor) + Long.BYTES * 2;
    }

    private static void assertConfigEquals(QwenConfig expected, QwenConfig actual) {
        assertEquals(expected.vocabSize(), actual.vocabSize());
        assertEquals(expected.hiddenSize(), actual.hiddenSize());
        assertEquals(expected.numHiddenLayers(), actual.numHiddenLayers());
        assertEquals(expected.numAttentionHeads(), actual.numAttentionHeads());
        assertEquals(expected.numKeyValueHeads(), actual.numKeyValueHeads());
        assertEquals(expected.attentionHeadDim(), actual.attentionHeadDim());
        assertEquals(expected.intermediateSize(), actual.intermediateSize());
        assertEquals(expected.linearNumKeyHeads(), actual.linearNumKeyHeads());
        assertEquals(expected.linearNumValueHeads(), actual.linearNumValueHeads());
        assertEquals(expected.linearKeyHeadDim(), actual.linearKeyHeadDim());
        assertEquals(expected.linearValueHeadDim(), actual.linearValueHeadDim());
        assertEquals(expected.linearConvKernelDim(), actual.linearConvKernelDim());
        assertEquals(expected.rmsNormEpsilon(), actual.rmsNormEpsilon());
        assertEquals(expected.ropeTheta(), actual.ropeTheta());
        assertEquals(expected.partialRotaryFactor(), actual.partialRotaryFactor());
        assertEquals(expected.maxPositionEmbeddings(), actual.maxPositionEmbeddings());
        assertEquals(expected.hiddenActivation(), actual.hiddenActivation());
        assertArrayEquals(expected.layerTypes(), actual.layerTypes());
        assertEquals(expected.numExperts(), actual.numExperts());
        assertEquals(expected.numExpertsPerToken(), actual.numExpertsPerToken());
        assertEquals(expected.moeIntermediateSize(), actual.moeIntermediateSize());
        assertEquals(expected.sharedExpertIntermediateSize(), actual.sharedExpertIntermediateSize());
        assertEquals(expected.tieWordEmbeddings(), actual.tieWordEmbeddings());
        assertEquals(expected.attentionOutputGate(), actual.attentionOutputGate());
        assertEquals(expected.mtpLayerCount(), actual.mtpLayerCount());
    }

    private static void assertTensorDescriptorsEqual(TensorDescriptor[] expected, TensorDescriptor[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i].name(), actual[i].name());
            assertArrayEquals(expected[i].shape(), actual[i].shape());
            assertEquals(expected[i].dataType(), actual[i].dataType());
            assertEquals(expected[i].format(), actual[i].format());
            assertEquals(expected[i].dataOffset(), actual[i].dataOffset());
            assertEquals(expected[i].byteSize(), actual[i].byteSize());
        }
    }
}
