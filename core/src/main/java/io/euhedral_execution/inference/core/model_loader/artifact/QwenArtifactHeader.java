package io.euhedral_execution.inference.core.model_loader.artifact;

/// The fixed 48-byte header of a version 1 Qwen artifact.
///
/// All fields are encoded in big-endian byte order. The on-disk reserved field is always zero and
/// is intentionally not part of this record.
public record QwenArtifactHeader(
        int magic,
        int version,
        long metadataOffset,
        long metadataSize,
        long tensorTableOffset,
        int tensorCount,
        long tensorDataOffset) {

    public static final int MAGIC = 0x5157454E;
    public static final int VERSION = 1;
    public static final int BYTE_SIZE = 48;
}
