package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.scheduling.frames.QwenInstructionFrame;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class QwenOperationFrameReuseTest {
    @Test
    void submissionErrorStillTerminatesTheRequestAtTheRealExecutorBoundary() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new ErrorAfterSubmissionGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var context = context(plan, 507);
        runner.submit(context);
        AbstractFrame frame = pullOne(runner);
        var receiver = new AtomicReference<LatticeReceiver>();
        new DefaultExecutor().input(new LatticeSource() {
            @Override
            public void addDownstream(LatticeReceiver downstream) {
                receiver.set(downstream);
            }

            @Override
            public long pull(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stop, long requested) {
                return 0;
            }

            @Override
            public void request(long requested) {}

            @Override
            public void complete() {}

            @Override
            public boolean isComplete() {
                return false;
            }
        });

        assertThrows(OutOfMemoryError.class, () -> receiver.get().push(frame));
        assertTrue(context.outcome().isDone(), "an Error must not strand the runner outside the executor catch");
        assertEquals(
                QwenExecutionContext.Status.FAILED, context.outcome().join().status());
        runner.completeGracefully();
        assertTrue(runner.isComplete());
    }

    private static final class ErrorAfterSubmissionGpu extends QwenExecutionFixtures.RecordingGpu {
        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void submit(Runnable operation) {
            operation.run();
            throw new OutOfMemoryError("injected post-launch host exhaustion");
        }
    }

    @Test
    void concreteOperationFramesHaveDedicatedTypesAndAreRecycled() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(
                        QwenExecutionFixtures.q3("projection-a", 64, 201),
                        QwenExecutionFixtures.q3("projection-b", 64, 202)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runner = new QwenExecutionRunner(plan, gpu);

        List<AbstractFrame> firstRun = runQuantum(runner, plan, 301);
        List<AbstractFrame> secondRun = runQuantum(runner, plan, 302);

        assertEquals(
                List.of(
                        "io.euhedral_execution.inference.core.scheduling.frames.EmbeddingFrame",
                        "io.euhedral_execution.inference.core.scheduling.frames.RmsNormFrame",
                        "io.euhedral_execution.inference.core.scheduling.frames.LinearFrame",
                        "io.euhedral_execution.inference.core.scheduling.frames.LinearFrame"),
                firstRun.stream().map(frame -> frame.getClass().getName()).toList());
        assertNotSame(firstRun.get(2), firstRun.get(3));
        for (int index = 0; index < firstRun.size(); index++) {
            assertSame(firstRun.get(index), secondRun.get(index), "frame kind " + index + " was not recycled");
        }
        assertEquals(
                List.of("embed", "norm", "linear:201", "linear:202", "embed", "norm", "linear:201", "linear:202"),
                gpu.operations);
        runner.completeGracefully();
    }

    @Test
    void failedOperationFrameIsClearedAndRecycled() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        RuntimeException failure = new IllegalStateException("injected embedding failure");
        gpu.afterEmbedding = () -> {
            throw failure;
        };
        var first = new QwenExecutionContext(
                plan, new QwenSequenceState(401), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        runner.submit(first);
        AbstractFrame failedFrame = pullOne(runner);
        assertThrows(RuntimeException.class, failedFrame::execute);
        failedFrame.doFinallyWithError(failure);
        assertEquals(QwenExecutionContext.Status.FAILED, first.outcome().join().status());

        gpu.afterEmbedding = () -> {};
        var second = new QwenExecutionContext(
                plan, new QwenSequenceState(402), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {2});
        runner.submit(second);
        AbstractFrame reused = pullOne(runner);

        assertSame(failedFrame, reused);
        reused.execute();
        reused.doFinally();
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, second.outcome().join().status());
        runner.completeGracefully();
    }

    @Test
    void deferredRegistrationFailureDrainsGpuAndFinalizesTheOwningFrame() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new FailingDeferredGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var first = context(plan, 501);
        runner.submit(first);
        AbstractFrame frame = pullOne(runner);

        frame.execute();
        frame.doFinally();

        assertEquals(1, gpu.synchronizations);
        assertEquals(QwenExecutionContext.Status.FAILED, first.outcome().join().status());
        var second = context(plan, 502);
        runner.submit(second);
        AbstractFrame reused = pullOne(runner);
        assertSame(frame, reused, "failure must recycle the old frame exactly once");
        reused.execute();
        reused.doFinally();
        assertEquals(QwenExecutionContext.Status.FAILED, second.outcome().join().status());
        runner.completeGracefully();
    }

    private static final class FailingDeferredGpu extends QwenExecutionFixtures.RecordingGpu {
        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
            throw new IllegalStateException("injected event registration failure");
        }
    }

    @Test
    void embeddingUploadLivesUntilAsyncCompletionFinalizesTheFrame() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new DeferredUploadGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var context = context(plan, 506);
        runner.submit(context);
        AbstractFrame frame = pullOne(runner);
        frame.execute();
        frame.doFinally();

        assertEquals(1, gpu.uploads);
        assertEquals(0, gpu.releases);
        assertTrue(!context.outcome().isDone());
        gpu.completed.run();
        assertEquals(1, gpu.releases);
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        runner.completeGracefully();
    }

    private static final class DeferredUploadGpu extends QwenExecutionFixtures.RecordingGpu {
        int uploads;
        int releases;
        Runnable completed;

        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public UploadBuffer allocateUploadBuffer(long bytes) {
            uploads++;
            UploadBuffer allocated = super.allocateUploadBuffer(bytes);
            return new UploadBuffer(allocated.segment(), () -> {
                releases++;
                allocated.close();
            });
        }

        @Override
        public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
            this.completed = completed;
        }
    }

    @Test
    void failedRecoveryPoisonsGpuAndNeverReleasesUnprovenBuffers() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new UnrecoverableGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var context = context(plan, 503);
        runner.submit(context);
        AbstractFrame frame = pullOne(runner);
        RuntimeException failure = assertThrows(RuntimeException.class, frame::execute);
        frame.doFinallyWithError(failure);

        assertEquals(
                QwenExecutionContext.Status.FAILED, context.outcome().join().status());
        assertTrue(gpu.poisoned);
        assertEquals(0, gpu.nativeFrees, "uncertain GPU work may still use every submitted buffer");
        assertEquals(0, gpu.hostReleases, "uncertain DMA may still read the pinned upload");
        assertThrows(IllegalStateException.class, () -> runner.submit(context(plan, 504)));
        runner.completeGracefully();
    }

    @Test
    void failedDeferredRegistrationAndRecoveryFailsTheRequestWithoutReleasingBuffers() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new UnrecoverableRegistrationGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var context = context(plan, 505);
        runner.submit(context);
        AbstractFrame frame = pullOne(runner);
        frame.execute();
        frame.doFinally();

        assertTrue(gpu.poisoned);
        assertEquals(
                QwenExecutionContext.Status.FAILED, context.outcome().join().status());
        assertEquals(0, gpu.nativeFrees);
        runner.completeGracefully();
    }

    private static class UnrecoverableGpu extends QwenExecutionFixtures.RecordingGpu {
        protected boolean poisoned;
        protected int nativeFrees;
        protected int hostReleases;

        @Override
        public boolean asynchronous() {
            return true;
        }

        @Override
        public void submit(Runnable operation) {
            operation.run();
            throw new IllegalStateException("post-launch host error");
        }

        @Override
        public void synchronize() {
            throw new IllegalStateException("recovery failed");
        }

        @Override
        public void free(long address) {
            if (poisoned) throw new IllegalStateException("poisoned GPU retains the allocation");
            nativeFrees++;
            super.free(address);
        }

        @Override
        public UploadBuffer allocateUploadBuffer(long bytes) {
            UploadBuffer allocated = super.allocateUploadBuffer(bytes);
            return new UploadBuffer(allocated.segment(), () -> {
                hostReleases++;
                allocated.close();
            });
        }

        @Override
        public boolean completionProven() {
            return !poisoned;
        }

        @Override
        public void poison(Throwable failure) {
            poisoned = true;
        }

        @Override
        public void ensureHealthy() {
            if (poisoned) throw new IllegalStateException("GPU engine is poisoned");
        }
    }

    private static final class UnrecoverableRegistrationGpu extends UnrecoverableGpu {
        @Override
        public void submit(Runnable operation) {
            operation.run();
        }

        @Override
        public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
            throw new IllegalStateException("injected event registration failure");
        }
    }

    @Test
    void cancellationAfterFrameMaterializationRecyclesUndeliveredFrame() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var cancelled = context(plan, 403);
        var cancelledOutcome = runner.submit(cancelled);
        AtomicReference<AbstractFrame> deferredFrame = new AtomicReference<>();
        assertEquals(
                0,
                runner.pull(
                        ignored -> {},
                        frame -> {
                            deferredFrame.set(frame);
                            return true;
                        },
                        1));

        cancelled.cancel();
        var next = context(plan, 404);
        runner.submit(next);
        List<AbstractFrame> admitted = new ArrayList<>();
        assertEquals(1, runner.pull(admitted::add, frame -> false, 1));
        assertSame(deferredFrame.get(), admitted.getFirst());
        assertEquals(
                QwenExecutionContext.Status.CANCELLED, cancelledOutcome.join().status());
        admitted.getFirst().execute();
        admitted.getFirst().doFinally();
        assertEquals(QwenExecutionContext.Status.SUCCESS, next.outcome().join().status());
        runner.completeGracefully();
    }

    @Test
    void concurrentSuccessfulFinalizersRecycleFramesForLaterQuanta() throws Exception {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var runner = new QwenExecutionRunner(plan, gpu);
        var generator = new QwenWorkGenerator(plan, gpu, runner, ignored -> {});
        var firstContexts = List.of(context(plan, 409), context(plan, 410));
        List<QwenInstructionFrame> firstFrames = publishAndPull(generator, runner, firstContexts, gpu);
        assertNotSame(firstFrames.getFirst(), firstFrames.getLast());
        firstFrames.forEach(QwenInstructionFrame::execute);

        finishConcurrently(firstFrames);
        for (QwenExecutionContext context : firstContexts) {
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().join().status());
        }

        var secondContexts = List.of(context(plan, 411), context(plan, 412));
        List<QwenInstructionFrame> secondFrames = publishAndPull(generator, runner, secondContexts, gpu);
        assertTrue(firstFrames.containsAll(secondFrames));
        assertTrue(secondFrames.containsAll(firstFrames));
        secondFrames.forEach(frame -> {
            frame.execute();
            frame.doFinally();
        });
        for (QwenExecutionContext context : secondContexts) {
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().join().status());
        }
        runner.completeGracefully();
    }

    private static List<QwenInstructionFrame> publishAndPull(
            QwenWorkGenerator generator,
            QwenExecutionRunner runner,
            List<QwenExecutionContext> contexts,
            QwenExecutionFixtures.RecordingGpu gpu) {
        for (QwenExecutionContext context : contexts) {
            context.begin(gpu);
            generator.start(context);
        }
        List<AbstractFrame> frames = new ArrayList<>(contexts.size());
        assertEquals(contexts.size(), runner.pull(frames::add, frame -> false, contexts.size()));
        return frames.stream().map(QwenInstructionFrame.class::cast).toList();
    }

    private static void finishConcurrently(List<QwenInstructionFrame> frames) throws Exception {
        runConcurrently(frames, QwenInstructionFrame::doFinally);
    }

    private static void runConcurrently(
            List<QwenInstructionFrame> frames, Consumer<? super QwenInstructionFrame> action) throws Exception {
        try (ExecutorService workers = Executors.newFixedThreadPool(frames.size())) {
            CountDownLatch ready = new CountDownLatch(frames.size());
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> returned = new ArrayList<>(frames.size());
            for (QwenInstructionFrame frame : frames) {
                returned.add(workers.submit(() -> {
                    ready.countDown();
                    start.await();
                    action.accept(frame);
                    return null;
                }));
            }
            ready.await();
            start.countDown();
            for (Future<?> future : returned) future.get();
        }
    }

    private static QwenExecutionContext context(QwenExecutionPlan plan, long sequenceId) {
        return new QwenExecutionContext(
                plan, new QwenSequenceState(sequenceId), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
    }

    private static List<AbstractFrame> runQuantum(QwenExecutionRunner runner, QwenExecutionPlan plan, long sequenceId) {
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(sequenceId), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        runner.submit(context);
        List<AbstractFrame> frames = new ArrayList<>();
        for (int instruction = 0; instruction < plan.instructions().size(); instruction++) {
            List<AbstractFrame> next = new ArrayList<>(1);
            assertEquals(1, runner.pull(next::add, frame -> false, 1));
            AbstractFrame frame = next.getFirst();
            frames.add(frame);
            frame.execute();
            frame.doFinally();
        }
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        return frames;
    }

    private static AbstractFrame pullOne(QwenExecutionRunner runner) {
        List<AbstractFrame> next = new ArrayList<>(1);
        assertEquals(1, runner.pull(next::add, frame -> false, 1));
        return next.getFirst();
    }
}
