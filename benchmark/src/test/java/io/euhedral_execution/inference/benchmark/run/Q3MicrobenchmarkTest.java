package io.euhedral_execution.inference.benchmark.run;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Q3MicrobenchmarkTest {
    @Test
    void rejectsSingleDocumentOutputBeforeOpeningGpu(@TempDir Path directory) throws Exception {
        Path config = directory.resolve("config.json");
        Files.writeString(config, """
                {"artifact":"missing.edrl","tokenizer":"missing-tokenizer","cudaLibrary":"missing.so","output":"screen.json"}
                """);
        var options = BenchmarkOptions.load(config, Instant.EPOCH);
        var failure = assertThrows(IllegalArgumentException.class, () -> Q3Microbenchmark.run(options, null, null));
        assertTrue(failure.getMessage().contains("JSONL output"));
    }

    @Test
    void errorDistributionCountsEveryOutputAndRejectsNonfiniteValues() {
        try (Arena arena = Arena.ofConfined()) {
            var expected = arena.allocate(4, 2);
            var actual = arena.allocate(4, 2);
            expected.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3f80);
            expected.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4000);
            actual.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3f80);
            actual.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4040);
            var errors = Q3Microbenchmark.errors(expected, actual);
            assertEquals(2L, errors.get("count"));
            assertEquals(1L, errors.get("different"));
            assertEquals(1.0, errors.get("maxAbsolute"));
            assertEquals(0.5, errors.get("meanAbsolute"));
            assertFalse(
                    Q3Microbenchmark.accepts(io.euhedral_execution.inference.core.gpu.Q3DispatchMode.PREFILL, errors));
            assertFalse(
                    Q3Microbenchmark.accepts(io.euhedral_execution.inference.core.gpu.Q3DispatchMode.DECODE, errors));
            actual.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4001);
            var rounded = Q3Microbenchmark.errors(expected, actual);
            assertTrue(
                    Q3Microbenchmark.accepts(io.euhedral_execution.inference.core.gpu.Q3DispatchMode.PREFILL, rounded));
            assertFalse(
                    Q3Microbenchmark.accepts(io.euhedral_execution.inference.core.gpu.Q3DispatchMode.DECODE, rounded));
            assertArrayEquals(new long[] {1, 0, 0, 0, 1, 0}, (long[]) errors.get("absoluteBins"));
            actual.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x7f80);
            assertThrows(IllegalStateException.class, () -> Q3Microbenchmark.errors(expected, actual));
        }
    }
}
