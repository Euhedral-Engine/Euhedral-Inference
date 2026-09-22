package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record QwenAttentionWeights(
        TensorHandle qProj,
        TensorHandle kProj,
        TensorHandle vProj,
        TensorHandle oProj,
        TensorHandle qNorm,
        TensorHandle kNorm)
        implements QwenMixerWeights {}
