package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class AsyncGpuCompletionSinkTest {

    @Test
    void asyncRuntimeAttachesASeparateCompletionSource() {
        List<LatticeSource> sources = new ArrayList<>();
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new AsyncGpu();

        var runtime = new EuhedralInferenceRuntime(sources::add, plan, gpu);

        assertEquals(1, sources.size(), "CUDA completion frames need a lattice-owned source");
        assertFalse(sources.getFirst().isComplete());
        assertTrue(gpu.asynchronous());
        runtime.disconnectRunner();
        assertFalse(sources.getFirst().isComplete(), "runner detach must not close a reusable GPU completion source");
        runtime.attachRunner();
        runtime.disconnectRunner();
        assertFalse(sources.getFirst().isComplete());
        runtime.closeCompletionSink();
        assertTrue(sources.getFirst().isComplete());
    }

    @Test
    void gpuSignalEnqueuesFrameButOnlyLatticeExecutionFinalizesIt() {
        List<LatticeSource> sources = new ArrayList<>();
        var gpu = new AsyncGpu();
        var runtime =
                new EuhedralInferenceRuntime(sources::add, new QwenExecutionPlan(QwenExecutionFixtures.weights()), gpu);
        var source = sources.getFirst();
        source.addDownstream(new LatticeReceiver() {
            @Override
            public void addUpstream(LatticeSource upstream) {}

            @Override
            public void push(AbstractFrame frame) {
                throw new AssertionError("unexpected push");
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
        var finished = new AtomicBoolean();
        assertEquals(0, source.pull(AbstractFrame::execute, ignored -> false, 1));

        gpu.signal(() -> finished.set(true));
        assertFalse(finished.get(), "CUDA callback must not finalize off the Euhedral lattice");
        assertEquals(1, source.pull(AbstractFrame::execute, ignored -> false, 1));
        assertTrue(finished.get());
        runtime.disconnectRunner();
    }

    @Test
    void finalSinkCloseRejectsNewRunnerAdmission() {
        List<LatticeSource> sources = new ArrayList<>();
        var runtime = new EuhedralInferenceRuntime(
                sources::add, new QwenExecutionPlan(QwenExecutionFixtures.weights()), new AsyncGpu());

        runtime.closeCompletionSink();

        assertTrue(sources.getFirst().isComplete());
        assertThrows(IllegalStateException.class, runtime::attachRunner);
    }

    private static final class AsyncGpu extends QwenExecutionFixtures.RecordingGpu {
        private Consumer<Runnable> completionFrames;

        @Override
        public void bindCompletionSink(Consumer<Runnable> completionFrames) {
            this.completionFrames = completionFrames;
        }

        void signal(Runnable completion) {
            this.completionFrames.accept(completion);
        }

        @Override
        public boolean asynchronous() {
            return true;
        }
    }
}
