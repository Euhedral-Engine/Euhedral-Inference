package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;

public record TensorDescriptor(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long dataOffset,
        long byteSize) {

    public TensorDescriptor(
            String name, long[] shape, TensorDataType dataType, WeightFormat format, long dataOffset, long byteSize) {
        this(name, shape, dataType, format, defaultLayout(format), dataOffset, byteSize);
    }

    private static WeightLayout defaultLayout(WeightFormat format) {
        return switch (format) {
            case Q3_G64_FP16, Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, W8_G32_FP16, Q4, Q5 ->
                WeightLayout.ROW_SPLIT_K128_V1;
            default -> WeightLayout.CONTIGUOUS_LE_V1;
        };
    }
}
