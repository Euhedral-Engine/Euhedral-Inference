package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// Four-way fused attention projection used by the compact MTP object inventory.
public record QwenCompactMtpAttentionWeights(
        TensorHandle queryKeyGateValue, TensorHandle queryNorm, TensorHandle keyNorm, TensorHandle output)
        implements QwenMixerWeights {}
