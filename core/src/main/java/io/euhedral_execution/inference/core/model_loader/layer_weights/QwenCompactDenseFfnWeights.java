package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// Fused gate/up and compact down projections used by the Q3 runtime inventory.
public record QwenCompactDenseFfnWeights(TensorHandle gateUp, TensorHandle down) implements QwenFfnWeights {}
