package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record QwenMtpWeights(
        TensorHandle embeddingNorm,
        TensorHandle hiddenNorm,
        TensorHandle projection,
        QwenLayerWeights layer,
        TensorHandle finalNorm) {}
