package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.impl.DefaultExecutor;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class QwenExecutionContextTest {
    @Test
    void cancellationCleanupFailureStillPublishesOutcomeAndCanBeRetried() throws Exception {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(901);
        var lease = sequence.claimExecution(0);
        var attempts = new AtomicInteger();
        sequence.setRecurrentState(lease, (AutoCloseable) () -> {
            if (attempts.incrementAndGet() == 1) throw new IllegalStateException("transient free failure");
        });
        sequence.releaseExecution(lease, 0);
        var context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(1);
        assertTrue(outcome.isDone(), "terminal cleanup failure stranded the generation future");
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                outcome.get(1, TimeUnit.SECONDS).status());
        assertFalse(sequence.isExecutionClaimed());
        sequence.complete();
        assertEquals(2, attempts.get());
        runner.completeGracefully();
        assertTrue(runner.isComplete());
    }

    @Test
    void directSourceRunsEmbeddingNormAndIndependentProjections() throws Exception {
        var weights = QwenExecutionFixtures.weights();
        var plan = new QwenExecutionPlan(
                weights,
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("first", 64, 201), QwenExecutionFixtures.q3("second", 128, 202)));
        assertEquals(List.of(2, 3), plan.successors(1));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(10);
        var context = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1, 2});
        List<Long> outputs = new ArrayList<>();
        var runner = new QwenExecutionRunner(plan, gpu, completed -> {
            outputs.add(completed.workspace().projectionAddress(0));
            outputs.add(completed.workspace().projectionAddress(1));
            assertTrue(sequence.isExecutionClaimed());
            assertFalse(completed.workspace().isClosed());
        });
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(4);

        assertEquals(
                QwenExecutionContext.Status.SUCCESS,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(List.of("embed", "norm", "linear:201", "linear:202"), gpu.operations);
        assertEquals(4, gpu.synchronizations);
        assertNotEquals(outputs.get(0), outputs.get(1));
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertFalse(gpu.frees.contains(QwenExecutionFixtures.MODEL_ADDRESS));
        assertTrue(context.workspace().isClosed());
        assertEquals(2, sequence.currentTokenPosition());
        runner.completeGracefully();
        assertTrue(runner.isComplete());
    }

    @Test
    void pullHonorsStopWithoutConsumingOrGeneratingFrame() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var runner = new QwenExecutionRunner(plan, new QwenExecutionFixtures.RecordingGpu());
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(11), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        runner.submit(context);
        List<AbstractFrame> pulled = new ArrayList<>();
        assertEquals(0, runner.pull(pulled::add, frame -> true, 1));
        assertTrue(pulled.isEmpty());
        assertEquals(1, runner.pull(pulled::add, frame -> false, 1));
        assertEquals(1, pulled.size());
        // The caller owns the pulled frame, including its execution and terminal notification.
        pulled.getFirst().execute();
        pulled.getFirst().doFinally();
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        runner.completeGracefully();
    }

    @Test
    void pullBulkDrainsReadyFramesWithoutBorrowingBeyondDemand() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var runner = new QwenExecutionRunner(plan, new QwenExecutionFixtures.RecordingGpu());
        var first = new QwenExecutionContext(
                plan, new QwenSequenceState(101), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var second = new QwenExecutionContext(
                plan, new QwenSequenceState(102), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {2});
        runner.submit(first);
        runner.submit(second);
        List<AbstractFrame> borrowed = new ArrayList<>();
        assertEquals(2, runner.pull(borrowed::add, frame -> false, 3));
        assertEquals(0, runner.pull(borrowed::add, frame -> false, 3));
        for (AbstractFrame frame : borrowed) {
            frame.execute();
            frame.doFinally();
        }
        assertEquals(QwenExecutionContext.Status.SUCCESS, first.outcome().join().status());
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, second.outcome().join().status());
        runner.completeGracefully();
    }

    @Test
    void requestDrainsOnlyWorkReadyDuringThatSynchronousCall() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var runner = new QwenExecutionRunner(plan, new QwenExecutionFixtures.RecordingGpu());
        var pushes = new AtomicInteger();
        runner.addDownstream(new LatticeReceiver() {
            @Override
            public void addUpstream(LatticeSource source) {}

            @Override
            public void push(AbstractFrame frame) {
                pushes.incrementAndGet();
                frame.execute();
                frame.doFinally();
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
        runner.request(1);
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(16), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var outcome = runner.submit(context);
        assertEquals(0, pushes.get());
        assertFalse(outcome.isDone());
        runner.request(1);
        assertEquals(1, pushes.get());
        assertEquals(QwenExecutionContext.Status.SUCCESS, outcome.join().status());
    }

    @Test
    void cancellationAfterEmbeddingStopsSuccessorsAndReleasesWorkspace() throws Exception {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(12), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        gpu.afterEmbedding = context::cancel;
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(3);
        assertEquals(
                QwenExecutionContext.Status.CANCELLED,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(List.of("embed"), gpu.operations);
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertTrue(context.workspace().isClosed());
    }

    @Test
    void linearFailureStopsOtherWorkAndDoesNotFreeWeights() throws Exception {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("first", 64, 201), QwenExecutionFixtures.q3("second", 64, 202)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var failure = new IllegalStateException("injected linear failure");
        gpu.linearFailure = failure;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(13), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(4);
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertSame(failure, outcome.get().failure());
        assertEquals(List.of("embed", "norm", "linear:201"), gpu.operations);
        assertEquals(gpu.allocations.size(), gpu.frees.size());
        assertFalse(gpu.frees.contains(QwenExecutionFixtures.MODEL_ADDRESS));
    }

    @Test
    void failedTemporaryFreeIsRetriedAtTerminalCleanup() throws Exception {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        gpu.freeFailures = 1;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(14), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(1);
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                outcome.get(5, TimeUnit.SECONDS).status());
        assertEquals(gpu.allocations.size(), gpu.frees.size());
    }

    @Test
    void failedWorkspaceFreeIsRetriedBeforePublishingOutcome() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        gpu.failAddressOnce = 1000;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(20), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(1);
        assertEquals(QwenExecutionContext.Status.FAILED, outcome.join().status());
        assertTrue(context.workspace().isClosed());
        assertEquals(gpu.allocations.size(), gpu.frees.size());
    }

    @Test
    void partiallyAllocatedWorkspaceRemainsOwnedWhenConstructionCleanupFails() {
        var plan = new QwenExecutionPlan(
                QwenExecutionFixtures.weights(),
                QwenExecutionFixtures.norm(),
                List.of(QwenExecutionFixtures.q3("projection", 64, 201)));
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        gpu.failAllocationAt = 2;
        gpu.freeFailures = 1;
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(103), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runner = new QwenExecutionRunner(plan, gpu);

        assertEquals(
                QwenExecutionContext.Status.FAILED,
                runner.submit(context).join().status());
        assertEquals(gpu.allocations, gpu.frees);
        assertTrue(context.workspace().isClosed());
        runner.completeGracefully();
    }

    @Test
    void cancellationBetweenAdmissionCheckAndLeaseClaimPublishesCancelled() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(104);
        var context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});

        context.begin(gpu, sequence::cancel);

        assertEquals(
                QwenExecutionContext.Status.CANCELLED, context.outcome().join().status());
        assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void rejectedTokenDoesNotPoisonUnclaimedSequence() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var sequence = new QwenSequenceState(15);
        var context = new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {
            QwenExecutionFixtures.VOCABULARY
        });
        var runner = new QwenExecutionRunner(plan, gpu);
        assertEquals(
                QwenExecutionContext.Status.FAILED,
                runner.submit(context).join().status());
        assertEquals(QwenSequenceState.TerminalState.ACTIVE, sequence.terminalState());
        assertTrue(gpu.allocations.isEmpty());
    }

    @Test
    void duplicateAdmissionDoesNotRegisterAnotherTerminalOwner() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var runner = new QwenExecutionRunner(plan, new QwenExecutionFixtures.RecordingGpu());
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(17), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var first = runner.submit(context);
        int terminalDependents = context.completion().getNumberOfDependents();
        assertThrows(IllegalStateException.class, () -> runner.submit(context));
        assertEquals(terminalDependents, context.completion().getNumberOfDependents());
        new DefaultExecutor().input(runner);
        runner.completeGracefully();
        runner.request(1);
        assertEquals(QwenExecutionContext.Status.SUCCESS, first.join().status());
        assertTrue(runner.isComplete());
    }

    @Test
    void callersCannotCompleteTheInternalQuantumOutcome() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(18), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var runner = new QwenExecutionRunner(plan, gpu);
        new DefaultExecutor().input(runner);
        var exposed = runner.submit(context);
        exposed.complete(new QwenExecutionContext.Outcome(QwenExecutionContext.Status.SUCCESS, null));
        assertFalse(context.outcome().isDone());
        runner.completeGracefully();
        assertFalse(runner.isComplete());
        runner.request(1);
        assertEquals(
                QwenExecutionContext.Status.SUCCESS, context.outcome().join().status());
        assertTrue(context.workspace().isClosed());
        assertTrue(runner.isComplete());
    }

    @Test
    void failureRecordedDuringTerminalConsumerCannotCommitSuccess() {
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.weights());
        var gpu = new QwenExecutionFixtures.RecordingGpu();
        var context = new QwenExecutionContext(
                plan, new QwenSequenceState(19), QwenExecutionContext.ExecutionKind.DECODE, 0, new int[] {1});
        var failure = new IllegalStateException("external quantum failure");
        var runner = new QwenExecutionRunner(plan, gpu, ignored -> context.fail(failure));
        new DefaultExecutor().input(runner);
        var outcome = runner.submit(context);
        runner.request(1);
        assertEquals(QwenExecutionContext.Status.FAILED, outcome.join().status());
        assertSame(failure, outcome.join().failure());
        assertEquals(
                QwenSequenceState.TerminalState.FAILED, context.sequenceState().terminalState());
    }
}
