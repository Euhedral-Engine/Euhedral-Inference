package io.euhedral_execution.inference.core;

import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import java.util.BitSet;
import java.util.Objects;

/// Immutable engine tuning for controlled experiments. Each field is an independent axis.
///
/// `workerProcessorIds` are the resolved logical processor IDs handed to Euhedral; the engine
/// validates them against [ProcessorTopology] before loading any model resource. Use
/// [WorkerProcessorSelection] to derive them from topology. The set is copied on input and output.
/// `prefillChunkTokens` bounds the prompt tokens submitted per prefill quantum and therefore the
/// per-quantum GPU workspace. It does not change the token sequence.
public record InferenceTuning(BitSet workerProcessorIds, int prefillChunkTokens) {
    public static final int DEFAULT_PREFILL_CHUNK_TOKENS = QwenGenerationSession.DEFAULT_PREFILL_CHUNK_TOKENS;

    public InferenceTuning {
        workerProcessorIds = (BitSet)
                Objects.requireNonNull(workerProcessorIds, "workerProcessorIds").clone();
        if (workerProcessorIds.isEmpty()) throw new IllegalArgumentException("workerProcessorIds must not be empty");
        if (prefillChunkTokens <= 0) throw new IllegalArgumentException("prefillChunkTokens must be positive");
    }

    /// Default tuning for the given workers: the engine's existing prefill chunk size.
    public static InferenceTuning defaults(BitSet workerProcessorIds) {
        return new InferenceTuning(workerProcessorIds, DEFAULT_PREFILL_CHUNK_TOKENS);
    }

    /// Default tuning for a resolved selection.
    public static InferenceTuning defaults(WorkerProcessorSelection workers) {
        return defaults(Objects.requireNonNull(workers, "workers").processorIds());
    }

    public InferenceTuning withWorkerProcessorIds(BitSet ids) {
        return new InferenceTuning(ids, this.prefillChunkTokens);
    }

    public InferenceTuning withPrefillChunkTokens(int tokens) {
        return new InferenceTuning(this.workerProcessorIds, tokens);
    }

    @Override
    public BitSet workerProcessorIds() {
        return (BitSet) this.workerProcessorIds.clone();
    }
}
