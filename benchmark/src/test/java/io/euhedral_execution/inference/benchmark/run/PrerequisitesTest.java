package io.euhedral_execution.inference.benchmark.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.benchmark.BenchmarkFixtures;
import io.euhedral_execution.inference.core.ProcessorTopology;
import java.util.List;
import org.junit.jupiter.api.Test;

class PrerequisitesTest {
    @Test
    void parsesProcessorIdListsAndRanges() {
        assertEquals(List.of(2, 3, 4, 8), Prerequisites.ids("2-4, 8", "cpus"));
        assertThrows(IllegalArgumentException.class, () -> Prerequisites.ids("5-2", "cpus"));
        assertThrows(IllegalArgumentException.class, () -> Prerequisites.ids("a", "cpus"));
    }

    @Test
    void cpuSelectionsResolveThroughTheCoreTopology() {
        var topology = ProcessorTopology.system();
        assertEquals(
                topology.allProcessors().processorIds(),
                Prerequisites.resolveWorkers("all", List.of(), List.of(), topology)
                        .processorIds());
        assertEquals(
                topology.onePerPhysicalCore().processorIds(),
                Prerequisites.resolveWorkers("one-per-core", List.of(), List.of(), topology)
                        .processorIds());
        assertEquals(
                topology.performanceCoreProcessors().onePerCore().processorIds(),
                Prerequisites.resolveWorkers("performance-one-per-core", List.of(), List.of(), topology)
                        .processorIds());
        int first = topology.availableProcessorIds().nextSetBit(0);
        assertEquals(
                BenchmarkFixtures.bits(first),
                Prerequisites.resolveWorkers(Integer.toString(first), List.of(), List.of(), topology)
                        .processorIds());
        assertThrows(
                IllegalArgumentException.class,
                () -> Prerequisites.resolveWorkers(Integer.toString(first), List.of(first), List.of(), topology));
        assertThrows(
                IllegalArgumentException.class,
                () -> Prerequisites.resolveWorkers("1048576", List.of(), List.of(), topology));
    }
}
