package io.euhedral_execution.inference.core.gpu;

/// Synchronous GPU operations required by the first Qwen execution stage.
public interface QwenExecutionGpu extends GpuMemory {

    void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize);

    void synchronize();
}
