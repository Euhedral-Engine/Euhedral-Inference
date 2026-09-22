package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record QwenMoeWeights(
        TensorHandle router,
        QwenExpertWeights[] experts,
        QwenDenseFfnWeights sharedExpert,
        TensorHandle sharedExpertGate)
        implements QwenFfnWeights {}
