package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class InferenceEngineTest {

    @Test
    void describeReturnsEngineName() {
        InferenceEngine engine = new InferenceEngine();
        assertEquals("Euhedral-Inference core engine", engine.describe());
    }
}
