package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record TensorHandle(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long deviceAddress,
        long byteSize) {

    public TensorHandle(
            String name,
            long[] shape,
            TensorDataType dataType,
            WeightFormat format,
            long deviceAddress,
            long byteSize) {
        this(name, shape, dataType, format, WeightLayout.CONTIGUOUS_LE_V1, deviceAddress, byteSize);
    }
}
