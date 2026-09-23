package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import java.lang.reflect.Method;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class QwenFirstLayerWeightLoaderTest {

    @Test
    void exposesSelectiveCompactFirstLayerLoading() {
        Method method = assertDoesNotThrow(() ->
                QwenWeightLoader.class.getMethod("loadFirstLayer", Path.class, QwenArtifact.class, GpuMemory.class));

        assertEquals(QwenWeights.class, method.getReturnType());
    }
}
