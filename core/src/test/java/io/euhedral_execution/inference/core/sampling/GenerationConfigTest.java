package io.euhedral_execution.inference.core.sampling;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class GenerationConfigTest {

    @Test
    void rejectsInvalidSamplingRanges() {
        assertThrows(IllegalArgumentException.class, () -> new GenerationConfig(Float.NaN, 0, 1.0f, 0L, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> new GenerationConfig(Float.POSITIVE_INFINITY, 0, 1.0f, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new GenerationConfig(1.0f, -1, 1.0f, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new GenerationConfig(1.0f, 0, 0.0f, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new GenerationConfig(1.0f, 0, Float.NaN, 0L, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> new GenerationConfig(1.0f, 0, Float.POSITIVE_INFINITY, 0L, false));
        assertThrows(IllegalArgumentException.class, () -> new GenerationConfig(1.0f, 0, 1.1f, 0L, false));
    }
}
