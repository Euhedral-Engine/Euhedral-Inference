package io.euhedral_execution.inference.api.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;

class InferencePropertiesTest {
    private static final Path PATH = Path.of("/x");

    @Test
    void parsesProcessorIdsAndInclusiveRanges() {
        BitSet expected = new BitSet();
        expected.set(2, 6);
        expected.set(8);
        assertEquals(expected, InferenceProperties.parseCpus(" 2-5, 8 ,"));
        var config = new InferenceProperties(PATH, PATH, PATH, "3", Duration.ofSeconds(4), "m").toInferenceConfig();
        assertEquals(BitSet.valueOf(new long[] {1L << 3}), config.workerCpus());
        assertEquals(Duration.ofSeconds(4), config.shutdownTimeout());
    }

    @Test
    void rejectsInvalidConfigurationAtBindTime() {
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus("5-2"));
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus("-1"));
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus("a"));
        assertThrows(IllegalArgumentException.class, () -> InferenceProperties.parseCpus(" , "));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceProperties(null, PATH, PATH, "1", Duration.ofSeconds(1), "m"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceProperties(PATH, PATH, PATH, "1", Duration.ofSeconds(1), " "));
    }
}
