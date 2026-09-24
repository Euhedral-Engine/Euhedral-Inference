package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.floatToBf16;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class QwenFullModelCpuReferenceTest {

    @Test
    void singleTokenAttentionMapsGroupedValuesAndAppliesPerQueryGate() {
        int queryHeads = 4;
        int keyValueHeads = 2;
        int headDim = 2;
        short[] gateValue = new short[] {
            floatToBf16(0.0f), floatToBf16(1.0f),
            floatToBf16(2.0f), floatToBf16(3.0f),
            floatToBf16(4.0f), floatToBf16(5.0f),
            floatToBf16(6.0f), floatToBf16(7.0f),
            floatToBf16(10.0f), floatToBf16(20.0f),
            floatToBf16(30.0f), floatToBf16(40.0f)
        };

        short[] output =
                QwenFullModelCpuReference.singleTokenAttentionContext(gateValue, queryHeads, keyValueHeads, headDim);

        for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
            int valueHead = queryHead / (queryHeads / keyValueHeads);
            for (int lane = 0; lane < headDim; lane++) {
                float gate = bf16ToFloat(gateValue[queryHead * headDim + lane]);
                float value = bf16ToFloat(gateValue[queryHeads * headDim + valueHead * headDim + lane]);
                float expected = value / (1.0f + (float) Math.exp(-gate));
                assertEquals(expected, bf16ToFloat(output[queryHead * headDim + lane]), 0.06f);
            }
        }
    }
}
