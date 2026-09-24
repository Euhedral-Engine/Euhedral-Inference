package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import java.io.IOException;

public final class EngineModelFixture {
    public static QwenModel load(GpuMemory gpu, QwenWeights weights) throws IOException {
        return QwenModel.load(gpu, memory -> {
            memory.allocate(128);
            return weights;
        });
    }
}
