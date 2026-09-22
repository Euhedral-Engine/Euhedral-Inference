package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record QwenLayerWeights(
        int index,
        TensorHandle inputNorm,
        TensorHandle postAttentionNorm,
        QwenMixerWeights mixer,
        QwenFfnWeights ffn) {}
