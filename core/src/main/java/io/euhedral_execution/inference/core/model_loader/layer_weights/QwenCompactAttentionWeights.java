package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// Fused full-attention runtime objects from the compact NInfer-compatible inventory.
public record QwenCompactAttentionWeights(
        TensorHandle queryKey,
        TensorHandle gateValue,
        TensorHandle queryNorm,
        TensorHandle keyNorm,
        TensorHandle output)
        implements QwenMixerWeights {}
