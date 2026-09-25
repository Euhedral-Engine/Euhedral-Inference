package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.core.frames.RunnableFrame;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class InferenceGpuExecutorTest {
    @Test
    void clonedWorkersBindTheirGpuIdentityForEachFrame() {
        var gpu = new RecordingGpu();
        var prototype = new InferenceGpuExecutor(gpu);
        var first = prototype.hookOnClone(3);
        var second = prototype.hookOnClone(7);
        assertEquals(List.of(3, 7), gpu.openedCpus);
        var ran = new ArrayList<Integer>();

        first.execute(new RunnableFrame(1L, () -> ran.add(1)));
        first.execute(new RunnableFrame(2L, () -> ran.add(2)));
        second.execute(new RunnableFrame(3L, () -> ran.add(3)));

        assertEquals(List.of(1, 2, 3), ran);
        assertEquals(List.of(3, 3, 7), gpu.boundCpus);
    }

    @Test
    void closingOwnerClosesBuddyAndBothWorkerStreams() {
        var gpu = new RecordingGpu();
        var primary = new InferenceGpuExecutor(gpu).hookOnClone(3);
        primary.hookOnClone(7);
        primary.close();
        assertEquals(List.of(7, 3), gpu.closedCpus);
    }

    @Test
    void retiringWorkerDoesNotCloseReplacementOnSameCpu() {
        var gpu = new RecordingGpu();
        var prototype = new InferenceGpuExecutor(gpu);
        var retired = prototype.hookOnClone(3);
        var replacement = prototype.hookOnClone(3);
        retired.close();
        replacement.execute(new RunnableFrame(4L, () -> {}));
        assertEquals(List.of(3), gpu.boundCpus);
    }

    private static final class RecordingGpu extends QwenExecutionFixtures.RecordingGpu {
        private final List<Integer> boundCpus = new ArrayList<>();
        private final List<Integer> openedCpus = new ArrayList<>();
        private final List<Integer> closedCpus = new ArrayList<>();
        private final Map<Long, Integer> workers = new HashMap<>();
        private final Set<Long> closedWorkers = new HashSet<>();
        private long nextWorker;

        @Override
        public long openWorker(int cpu) {
            openedCpus.add(cpu);
            long worker = ++nextWorker;
            workers.put(worker, cpu);
            return worker;
        }

        @Override
        public void closeWorker(long worker) {
            closedCpus.add(workers.get(worker));
            closedWorkers.add(worker);
        }

        @Override
        public void withWorker(long worker, Runnable operation) {
            if (closedWorkers.contains(worker)) throw new IllegalStateException("worker stream was closed");
            boundCpus.add(workers.get(worker));
            operation.run();
        }
    }
}
