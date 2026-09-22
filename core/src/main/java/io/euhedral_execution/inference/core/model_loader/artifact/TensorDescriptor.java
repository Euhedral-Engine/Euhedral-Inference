package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;

public record TensorDescriptor(
        String name, long[] shape, TensorDataType dataType, WeightFormat format, long dataOffset, long byteSize) {}
