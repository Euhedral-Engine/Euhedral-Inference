package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record TensorHandle(
        String name, long[] shape, TensorDataType dataType, WeightFormat format, long deviceAddress, long byteSize) {}
