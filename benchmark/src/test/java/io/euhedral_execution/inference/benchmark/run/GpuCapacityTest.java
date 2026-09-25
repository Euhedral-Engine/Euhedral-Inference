package io.euhedral_execution.inference.benchmark.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

class GpuCapacityTest {
    private static final long GIB = 1L << 30;
    private static final long ARTIFACT_BYTES = 12_859_040_768L;
    private static final long DEFAULT_HEADROOM_BYTES = 1024L * 1024 * 1024;

    @Test
    void refusesWhenAnotherProcessHoldsTheMemoryTheModelNeeds() {
        // The development host's situation: a serving container holds most of a 16 GiB device.
        assertEquals(
                "GPU has 2.1 GiB free of 15.9 GiB; loading needs 13.0 GiB (artifact plus gpuHeadroomMiB)",
                GpuCapacity.evaluate(2_250_000_000L, 17_094_475_776L, ARTIFACT_BYTES + DEFAULT_HEADROOM_BYTES));
    }

    @Test
    void allowsLoadingWhenFreeMemoryCoversArtifactAndHeadroom() {
        assertNull(GpuCapacity.evaluate(14 * GIB, 16 * GIB, 13 * GIB));
        assertNull(GpuCapacity.evaluate(13 * GIB, 16 * GIB, 13 * GIB), "exactly enough is enough");
    }
}
