package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record QwenDenseFfnWeights(TensorHandle gateProj, TensorHandle upProj, TensorHandle downProj)
        implements QwenFfnWeights {}
