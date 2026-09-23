package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// Persistent byte layout independent of the source tensor data type.
public enum WeightLayout {
    CONTIGUOUS_LE_V1,
    ROW_SPLIT_K128_V1
}
