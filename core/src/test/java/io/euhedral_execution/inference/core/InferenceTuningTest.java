package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;

class InferenceTuningTest {
    private static final Path PATH = Path.of("/x");

    @Test
    void defaultPrefillChunkIsTheExistingSessionValue() {
        assertEquals(512, InferenceTuning.DEFAULT_PREFILL_CHUNK_TOKENS);
        assertEquals(512, QwenGenerationSession.DEFAULT_PREFILL_CHUNK_TOKENS);
        assertEquals(512, InferenceTuning.defaults(ProcessorTopology.bits(3)).prefillChunkTokens());
    }

    @Test
    void legacyConfigConstructionUsesDefaultTuningWithOneWorkerSet() {
        BitSet cpus = ProcessorTopology.bits(2, 5);
        var legacy = new InferenceConfig(PATH, PATH, PATH, cpus, Duration.ofSeconds(1));
        assertEquals(InferenceTuning.defaults(cpus), legacy.tuning());
        assertEquals(cpus, legacy.workerCpus());
        assertEquals(legacy.tuning().workerProcessorIds(), legacy.workerCpus());
        assertEquals(
                legacy, new InferenceConfig(PATH, PATH, PATH, InferenceTuning.defaults(cpus), Duration.ofSeconds(1)));
    }

    @Test
    void programmaticConfigCarriesTuning() {
        var tuning = new InferenceTuning(ProcessorTopology.bits(4), 128);
        var config = new InferenceConfig(PATH, PATH, PATH, tuning, Duration.ofSeconds(1));
        assertEquals(tuning, config.tuning());
        assertEquals(ProcessorTopology.bits(4), config.workerCpus());
    }

    @Test
    void validatesEveryAxis() {
        assertThrows(IllegalArgumentException.class, () -> new InferenceTuning(new BitSet(), 512));
        assertThrows(NullPointerException.class, () -> new InferenceTuning(null, 512));
        assertThrows(IllegalArgumentException.class, () -> new InferenceTuning(ProcessorTopology.bits(1), 0));
        assertThrows(IllegalArgumentException.class, () -> new InferenceTuning(ProcessorTopology.bits(1), -1));
        assertThrows(
                NullPointerException.class,
                () -> new InferenceConfig(PATH, PATH, PATH, (InferenceTuning) null, Duration.ofSeconds(1)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceConfig(PATH, PATH, PATH, new BitSet(), Duration.ofSeconds(1)));
    }

    @Test
    void copiesWorkerIdsOnInputAndOutput() {
        BitSet input = ProcessorTopology.bits(1, 7);
        var tuning = new InferenceTuning(input, 64);
        input.set(9);
        tuning.workerProcessorIds().clear();
        assertEquals(ProcessorTopology.bits(1, 7), tuning.workerProcessorIds());
        var config = new InferenceConfig(PATH, PATH, PATH, tuning, Duration.ofSeconds(1));
        config.workerCpus().clear();
        assertEquals(ProcessorTopology.bits(1, 7), config.tuning().workerProcessorIds());
    }

    @Test
    void witherReturnsANewValueWithoutChangingTheOriginal() {
        var original = InferenceTuning.defaults(ProcessorTopology.bits(1));
        var changed = original.withPrefillChunkTokens(32).withWorkerProcessorIds(ProcessorTopology.bits(2));
        assertEquals(512, original.prefillChunkTokens());
        assertEquals(ProcessorTopology.bits(1), original.workerProcessorIds());
        assertEquals(new InferenceTuning(ProcessorTopology.bits(2), 32), changed);
        assertNotEquals(original, changed);
        assertEquals(
                InferenceTuning.defaults(ProcessorTopology.bits(0, 8)),
                InferenceTuning.defaults(WorkerProcessorSelectionTest.HYBRID.processors(0, 8)));
    }
}
