package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import java.util.Objects;

/// Persistent per-layer key/value caches for one Qwen sequence.
public final class AttentionSequenceStates implements AutoCloseable {

    private final AttentionKvState[] states;
    private boolean closed;

    private AttentionSequenceStates(AttentionKvState[] states) {
        this.states = states;
    }

    public static AttentionSequenceStates allocate(ExecutionGpu gpu, QwenLayerType[] layerTypes, int keyValueWidth) {
        Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(layerTypes, "layerTypes");
        AttentionKvState[] states = new AttentionKvState[layerTypes.length];
        for (int index = 0; index < layerTypes.length; index++) {
            if (layerTypes[index] == QwenLayerType.FULL_ATTENTION) {
                states[index] = new AttentionKvState(gpu, keyValueWidth);
            }
        }
        return new AttentionSequenceStates(states);
    }

    public AttentionKvState forLayer(int layerIndex) {
        if (this.closed) throw new IllegalStateException("attention sequence states are closed");
        if (layerIndex < 0 || layerIndex >= this.states.length || this.states[layerIndex] == null) {
            throw new IllegalArgumentException("layer does not have full-attention KV state: " + layerIndex);
        }
        return this.states[layerIndex];
    }

    @Override
    public void close() {
        if (this.closed) return;
        Throwable failure = null;
        for (int index = this.states.length - 1; index >= 0; index--) {
            AttentionKvState state = this.states[index];
            if (state == null) continue;
            try {
                state.close();
                this.states[index] = null;
            } catch (Throwable cleanupFailure) {
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
        }
        this.closed = true;
        for (AttentionKvState state : this.states) this.closed &= state == null;
        if (failure instanceof Error error) throw error;
        if (failure instanceof RuntimeException runtimeException) throw runtimeException;
        if (failure != null) throw new IllegalStateException("failed to release attention sequence states", failure);
    }
}
