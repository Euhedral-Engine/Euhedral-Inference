package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// Fused GDN runtime objects from the compact NInfer-compatible inventory.
public record QwenCompactGatedDeltaNetWeights(
        TensorHandle aLog,
        TensorHandle dtBias,
        TensorHandle convolution,
        TensorHandle aProjection,
        TensorHandle bProjection,
        TensorHandle queryKey,
        TensorHandle valueZ,
        TensorHandle norm,
        TensorHandle output)
        implements QwenMixerWeights {}
