package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class AttentionKvStateTest {

    @Test
    void growsCacheWithoutDroppingPriorKeysOrValuesAndReleasesAllocations() {
        RecordingGpu gpu = new RecordingGpu();
        AttentionKvState state = new AttentionKvState(gpu, 4);

        state.prepareAppend(0, 3);
        long original = state.keyCacheAddress();
        long originalValue = state.valueCacheAddress();
        assertEquals(3, state.capacity());
        state.commitAppend(3);

        state.prepareAppend(3, 2);
        long grown = state.keyCacheAddress();
        assertTrue(grown != original);
        assertTrue(state.valueCacheAddress() != originalValue);
        assertEquals(6, state.capacity());
        assertEquals(3, state.length());
        assertEquals(List.of(24L, 24L), gpu.copySizes);
        state.commitAppend(2);
        assertEquals(5, state.length());

        state.close();
        assertEquals(2, gpu.frees.size());
        assertTrue(gpu.frees.contains(original));
        assertTrue(gpu.frees.contains(grown));
    }

    @Test
    void asynchronousGrowthRetainsOldCacheUntilAppendCompletion() {
        RecordingGpu gpu = new RecordingGpu(true);
        AttentionKvState state = new AttentionKvState(gpu, 4);
        state.prepareAppend(0, 3);
        long original = state.keyCacheAddress();
        state.commitAppend(3);

        state.prepareAppend(3, 2);
        assertTrue(state.keyCacheAddress() != original);
        assertTrue(gpu.frees.isEmpty(), "queued copies must retain the source allocation");
        state.commitAppend(2);
        assertEquals(List.of(original), gpu.frees);
        state.close();
    }

    @Test
    void rejectsGapsAndDoesNotAdvanceLengthBeforeAppendCommit() {
        RecordingGpu gpu = new RecordingGpu();
        AttentionKvState state = new AttentionKvState(gpu, 4);

        assertThrows(IllegalArgumentException.class, () -> state.prepareAppend(1, 1));
        state.prepareAppend(0, 2);
        assertEquals(0, state.length());
        state.commitAppend(2);
        assertThrows(IllegalArgumentException.class, () -> state.prepareAppend(1, 1));
        assertEquals(2, state.length());
        state.close();
    }

    private static final class RecordingGpu extends ExecutionGpu {
        private final boolean asynchronous;
        private final AtomicLong nextAddress = new AtomicLong(1000);
        private final List<Long> copySizes = new ArrayList<>();
        private final List<Long> frees = new ArrayList<>();

        private RecordingGpu() {
            this(false);
        }

        private RecordingGpu(boolean asynchronous) {
            this.asynchronous = asynchronous;
        }

        @Override
        public boolean asynchronous() {
            return asynchronous;
        }

        @Override
        public long allocate(long byteSize) {
            return nextAddress.getAndIncrement();
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void free(long address) {
            frees.add(address);
        }

        @Override
        public void copyDeviceToDevice(long destination, long source, long byteSize) {
            copySizes.add(byteSize);
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {}

        @Override
        public void synchronize() {}
    }
}
