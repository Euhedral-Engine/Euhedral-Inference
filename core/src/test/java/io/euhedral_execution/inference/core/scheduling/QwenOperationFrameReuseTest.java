package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.scheduling.frames.QwenInstructionFrame;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class QwenOperationFrameReuseTest {

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
