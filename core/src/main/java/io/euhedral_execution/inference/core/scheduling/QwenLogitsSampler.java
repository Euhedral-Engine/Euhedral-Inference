package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.sampling.TokenSampler;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.IntPredicate;

/// Bridges retained Qwen BF16 logits to the independent host-side token sampler.
/// Keep one instance per generation so its seeded random state is request-local.
public final class QwenLogitsSampler {

    private final TokenSampler sampler;

    public QwenLogitsSampler(GenerationConfig config, int vocabularySize) {
        this.sampler = new TokenSampler(config, vocabularySize);
    }

    /// Copies only the final vocabulary row and does not close or otherwise claim the logits allocation.
    public int selectToken(QwenDeviceLogits logits, ExecutionGpu gpu) {
        return selectToken(logits, gpu, null);
    }

    /// Applies a request-local vocabulary constraint before temperature, top-k, or top-p sampling.
    public int selectToken(QwenDeviceLogits logits, ExecutionGpu gpu, IntPredicate allowed) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(gpu, "gpu");
        int vocabularySize = logits.vocabularySize();
        if (vocabularySize != this.sampler.vocabularySize()) {
            throw new IllegalArgumentException(
                    "sampler expects " + this.sampler.vocabularySize() + " logits, got " + vocabularySize);
        }

        long rowByteSize = Math.multiplyExact((long) vocabularySize, Short.BYTES);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment row = arena.allocate(rowByteSize, Short.BYTES);
            logits.copyFinalTokenRowToHost(gpu, row);
            float[] hostLogits = new float[vocabularySize];
            for (int tokenId = 0; tokenId < vocabularySize; tokenId++) {
                short bf16Bits = row.get(ValueLayout.JAVA_SHORT, (long) tokenId * Short.BYTES);
                hostLogits[tokenId] = allowed != null && !allowed.test(tokenId)
                        ? Float.NEGATIVE_INFINITY
                        : Float.intBitsToFloat(Short.toUnsignedInt(bf16Bits) << 16);
            }
            return this.sampler.selectToken(hostLogits);
        }
    }
}
