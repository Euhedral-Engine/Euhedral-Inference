package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.PipelineFrame;
import io.euhedral_execution.core.generics.AbstractExecutor;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class QwenExecutionPlanTest {

    private static final long TIMEOUT_SECONDS = 5;

    @Test
    void oneSubmissionExecutesPreparationLayersAndTerminalInModelOrder() throws Exception {
        QwenExecutionPlan plan =
                new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION, QwenLayerType.GATED_DELTA_NET));
        QwenSequenceState sequence = new QwenSequenceState(11L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0L, 3);
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(10);

            assertEquals(
                    PipelineFrame.Status.SUCCESS,
                    outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS).status());
            assertEquals(
                    List.of(
                            "prepare/embed",
                            "layer[0]:FULL_ATTENTION",
                            "layer[1]:GATED_DELTA_NET",
                            "final-norm/lm-head",
                            "terminal-result"),
                    context.executionTrace());
            assertEquals(3L, sequence.currentTokenPosition());
            assertFalse(sequence.isExecutionClaimed());
            assertEquals(QwenSequenceState.TerminalState.ACTIVE, sequence.terminalState());
        }
    }

    @Test
    void mixedLayerTopologyProducesPreselectedOperations() {
        QwenExecutionPlan plan = new QwenExecutionPlan(
                weights(QwenLayerType.FULL_ATTENTION, QwenLayerType.GATED_DELTA_NET, QwenLayerType.FULL_ATTENTION));

        assertEquals(
                List.of(
                        new QwenExecutionPlan.LayerOperation(0, QwenLayerType.FULL_ATTENTION),
                        new QwenExecutionPlan.LayerOperation(1, QwenLayerType.GATED_DELTA_NET),
                        new QwenExecutionPlan.LayerOperation(2, QwenLayerType.FULL_ATTENTION)),
                plan.layerOperations());
        assertEquals(3, plan.layerOperations().size());
    }

    @Test
    void planSnapshotsMutableWeightTopology() {
        QwenWeights sourceWeights = weights(QwenLayerType.FULL_ATTENTION);
        QwenExecutionPlan plan = new QwenExecutionPlan(sourceWeights);

        sourceWeights.layers()[0] = null;
        sourceWeights.config().layerTypes()[0] = QwenLayerType.GATED_DELTA_NET;

        assertNotSame(sourceWeights.layers(), plan.weights().layers());
        assertSame(plan.weights(), plan.weights());
        assertEquals(
                List.of(new QwenExecutionPlan.LayerOperation(0, QwenLayerType.FULL_ATTENTION)), plan.layerOperations());
        assertEquals(QwenLayerType.FULL_ATTENTION, plan.weights().config().layerTypes()[0]);
        assertTrue(plan.weights().layers()[0] != null);
        assertThrows(
                UnsupportedOperationException.class, () -> plan.layerWeights().clear());
    }

    @Test
    void independentContextsHaveIndependentTracesAndSequenceState() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState firstSequence = new QwenSequenceState(21L);
        QwenSequenceState secondSequence = new QwenSequenceState(22L);
        QwenExecutionContext first =
                new QwenExecutionContext(plan, firstSequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        QwenExecutionContext second =
                new QwenExecutionContext(plan, secondSequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 2);
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> firstOutcome = runner.submit(first);
            CompletableFuture<PipelineFrame.Outcome> secondOutcome = runner.submit(second);
            driver.request(20);

            assertEquals(
                    PipelineFrame.Status.SUCCESS,
                    firstOutcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS).status());
            assertEquals(
                    PipelineFrame.Status.SUCCESS,
                    secondOutcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS).status());
            assertNotSame(first.sequenceState(), second.sequenceState());
            assertEquals(1L, firstSequence.currentTokenPosition());
            assertEquals(2L, secondSequence.currentTokenPosition());
            assertTrue(first.executionTrace().stream().noneMatch(entry -> entry.contains("GATED_DELTA_NET")));
            assertEquals(first.executionTrace(), first.result().trace());
            assertEquals(second.executionTrace(), second.result().trace());
        }
    }

    @Test
    void cancelledContextProducesCancelledOutcomeAndTerminalState() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState sequence = new QwenSequenceState(31L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        context.cancel();
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(1);

            PipelineFrame.Outcome completed = outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.CANCELLED, completed.status());
            assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
            assertTrue(context.executionTrace().isEmpty());
        }
    }

    @Test
    void failedContextProducesFailedOutcomeWithOriginalFailure() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState sequence = new QwenSequenceState(41L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        IllegalStateException failure = new IllegalStateException("synthetic execution failure");
        context.fail(failure);
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(1);

            PipelineFrame.Outcome completed = outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertSame(failure, completed.failure());
            assertEquals(QwenSequenceState.TerminalState.FAILED, sequence.terminalState());
        }
    }

    @Test
    void runnerRejectsContextBelongingToAnotherPlan() throws Exception {
        QwenExecutionPlan runnerPlan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenExecutionPlan contextPlan =
                new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION, QwenLayerType.GATED_DELTA_NET));
        QwenSequenceState sequence = new QwenSequenceState(45L);
        QwenExecutionContext context =
                new QwenExecutionContext(contextPlan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        QwenExecutionRunner runner = runnerPlan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(1);

            PipelineFrame.Outcome completed = outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertTrue(completed.failure() instanceof IllegalArgumentException);
            assertTrue(context.executionTrace().isEmpty());
            assertEquals(QwenSequenceState.TerminalState.ACTIVE, sequence.terminalState());
        }
    }

    @Test
    void terminalConsumerFailureInvalidatesCapturedResultAndSequence() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState sequence = new QwenSequenceState(46L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        IllegalStateException failure = new IllegalStateException("terminal consumer failure");
        QwenExecutionRunner runner = plan.newRunner(ignored -> {
            throw failure;
        });

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(10);

            PipelineFrame.Outcome completed = outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertSame(failure, completed.failure());
            assertNull(context.result());
            assertEquals(QwenSequenceState.TerminalState.FAILED, sequence.terminalState());
            assertSame(failure, sequence.terminalFailure());
        }
    }

    @Test
    void terminalConsumerRunsBeforeSequenceLeaseRelease() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState sequence = new QwenSequenceState(48L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        QwenExecutionRunner runner = plan.newRunner(ignored -> {
            assertTrue(sequence.isExecutionClaimed());
            assertThrows(IllegalStateException.class, () -> sequence.claimExecution(1L));
        });

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            CompletableFuture<Void> finalized = outcome.thenRun(() -> {
                assertFalse(sequence.isExecutionClaimed());
                assertEquals(1L, sequence.currentTokenPosition());
            });
            driver.request(10);

            assertEquals(
                    PipelineFrame.Status.SUCCESS,
                    outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS).status());
            finalized.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertFalse(sequence.isExecutionClaimed());
            assertEquals(1L, sequence.currentTokenPosition());
        }
    }

    @Test
    void cancellationRequestedByTerminalConsumerProducesCancelledOutcome() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState sequence = new QwenSequenceState(49L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        QwenExecutionRunner runner = plan.newRunner(QwenExecutionContext::cancel);

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(10);

            assertEquals(
                    PipelineFrame.Status.CANCELLED,
                    outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS).status());
            assertNull(context.result());
            assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
            assertFalse(sequence.isExecutionClaimed());
        }
    }

    @Test
    void terminalConsumerErrorStillCompletesFailedOutcome() throws Exception {
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(QwenLayerType.FULL_ATTENTION));
        QwenSequenceState sequence = new QwenSequenceState(47L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.DECODE, 0L, 1);
        AssertionError failure = new AssertionError("terminal consumer error");
        QwenExecutionRunner runner = plan.newRunner(ignored -> {
            throw failure;
        });

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(10);

            PipelineFrame.Outcome completed = outcome.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertSame(failure, completed.failure().getCause());
            assertNull(context.result());
            assertEquals(QwenSequenceState.TerminalState.FAILED, sequence.terminalState());
            assertSame(failure, sequence.terminalFailure());
        }
    }

    @Test
    void sequenceRejectsConcurrentExecutionClaim() {
        QwenSequenceState sequence = new QwenSequenceState(51L);

        QwenSequenceState.ExecutionLease lease = sequence.claimExecution(0L);

        assertThrows(IllegalStateException.class, () -> sequence.claimExecution(0L));
        assertTrue(sequence.isExecutionClaimed());
        sequence.setKvCacheState(lease, "kv");
        sequence.setRecurrentState(lease, "recurrent");
        assertEquals("kv", sequence.kvCacheState());
        assertEquals("recurrent", sequence.recurrentState());
        sequence.releaseExecution(lease, 1L);
        assertFalse(sequence.isExecutionClaimed());
        assertEquals(1L, sequence.currentTokenPosition());
        assertThrows(IllegalStateException.class, () -> sequence.setKvCacheState(lease, "stale"));
        assertThrows(IllegalStateException.class, () -> sequence.releaseExecution(lease, 2L));
    }

    @Test
    void sequenceStateMethodsDoNotAcquireIntrinsicMonitors() {
        for (var method : QwenSequenceState.class.getDeclaredMethods()) {
            assertFalse(
                    Modifier.isSynchronized(method.getModifiers()),
                    () -> "QwenSequenceState method uses an intrinsic monitor: " + method.getName());
        }
    }

    @Test
    void concurrentClaimersPublishExactlyOneExecutionLease() throws Exception {
        int contenders = 24;
        QwenSequenceState sequence = new QwenSequenceState(52L);
        ExecutorService executor = Executors.newFixedThreadPool(contenders);
        CountDownLatch ready = new CountDownLatch(contenders);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<QwenSequenceState.ExecutionLease>> claims = new java.util.ArrayList<>(contenders);
        try {
            for (int index = 0; index < contenders; index++) {
                claims.add(executor.submit(() -> {
                    ready.countDown();
                    if (!start.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("claim race did not start");
                    }
                    try {
                        return sequence.claimExecution(0L);
                    } catch (IllegalStateException alreadyClaimed) {
                        return null;
                    }
                }));
            }
            assertTrue(ready.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            start.countDown();
            int winners = 0;
            QwenSequenceState.ExecutionLease winner = null;
            for (Future<QwenSequenceState.ExecutionLease> claim : claims) {
                QwenSequenceState.ExecutionLease lease = claim.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
                if (lease != null) {
                    winners++;
                    winner = lease;
                }
            }
            assertEquals(1, winners);
            sequence.releaseExecution(winner, 1L);
            assertEquals(1L, sequence.currentTokenPosition());
        } finally {
            start.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void cancellationBeforeLeaseReleasePreventsTokenPositionPublication() {
        QwenSequenceState sequence = new QwenSequenceState(54L);
        QwenSequenceState.ExecutionLease lease = sequence.claimExecution(0L);

        sequence.cancel();

        assertTrue(sequence.releaseExecutionAndCheckCancellation(lease, 1L));
        assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
        assertFalse(sequence.isExecutionClaimed());
        assertEquals(0L, sequence.currentTokenPosition());
    }

    @Test
    void cancellationRacingWithClaimIsResolvedByOneAtomicStateTransition() throws Exception {
        QwenSequenceState sequence = new QwenSequenceState(53L);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch start = new CountDownLatch(1);
        try {
            Future<QwenSequenceState.ExecutionLease> claim = executor.submit(() -> {
                assertTrue(start.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                try {
                    return sequence.claimExecution(0L);
                } catch (IllegalStateException cancellationWon) {
                    return null;
                }
            });
            Future<Void> cancel = executor.submit((Callable<Void>) () -> {
                assertTrue(start.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                sequence.cancel();
                return null;
            });
            start.countDown();

            QwenSequenceState.ExecutionLease lease = claim.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            cancel.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertTrue(sequence.cancellationRequested());
            if (lease != null) {
                assertTrue(sequence.releaseExecutionAndCheckCancellation(lease, 1L));
            }
            assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
            assertFalse(sequence.isExecutionClaimed());
            assertEquals(0L, sequence.currentTokenPosition());
        } finally {
            start.countDown();
            executor.shutdownNow();
        }
    }

    private static QwenWeights weights(QwenLayerType... layerTypes) {
        QwenLayerWeights[] layers = new QwenLayerWeights[layerTypes.length];
        for (int index = 0; index < layerTypes.length; index++) {
            layers[index] = layerTypes[index] == QwenLayerType.FULL_ATTENTION
                    ? new QwenLayerWeights(
                            index,
                            handle("layer-" + index + ".input"),
                            handle("layer-" + index + ".post"),
                            new QwenAttentionWeights(
                                    handle("layer-" + index + ".q"),
                                    handle("layer-" + index + ".k"),
                                    handle("layer-" + index + ".v"),
                                    handle("layer-" + index + ".o"),
                                    handle("layer-" + index + ".qn"),
                                    handle("layer-" + index + ".kn")),
                            new QwenDenseFfnWeights(
                                    handle("layer-" + index + ".gate"),
                                    handle("layer-" + index + ".up"),
                                    handle("layer-" + index + ".down")))
                    : new QwenLayerWeights(
                            index,
                            handle("layer-" + index + ".input"),
                            handle("layer-" + index + ".post"),
                            new QwenGatedDeltaNetWeights(
                                    handle("layer-" + index + ".qkv"),
                                    handle("layer-" + index + ".z"),
                                    handle("layer-" + index + ".b"),
                                    handle("layer-" + index + ".a"),
                                    handle("layer-" + index + ".conv"),
                                    handle("layer-" + index + ".norm"),
                                    handle("layer-" + index + ".dt"),
                                    handle("layer-" + index + ".log"),
                                    handle("layer-" + index + ".out")),
                            new QwenDenseFfnWeights(
                                    handle("layer-" + index + ".gate"),
                                    handle("layer-" + index + ".up"),
                                    handle("layer-" + index + ".down")));
        }
        return new QwenWeights(
                config(layerTypes), handle("embedding"), layers, handle("final-norm"), handle("lm-head"), null);
    }

    private static QwenConfig config(QwenLayerType[] layerTypes) {
        return new QwenConfig(
                32_000,
                4_096,
                layerTypes.length,
                32,
                8,
                128,
                11_008,
                16,
                8,
                64,
                128,
                4,
                1.0e-6,
                1_000_000.0,
                1.0,
                32_768,
                "silu",
                layerTypes.clone(),
                0,
                0,
                0,
                0,
                true,
                false,
                0);
    }

    private static TensorHandle handle(String name) {
        return new TensorHandle(name, new long[] {1}, TensorDataType.FP32, WeightFormat.FP32, 1L, 4L);
    }

    private static final class RunnerDriver implements AutoCloseable {

        private final QwenExecutionRunner runner;
        private final AbstractExecutor executor = new io.euhedral_execution.core.impl.DefaultExecutor();

        private RunnerDriver(QwenExecutionRunner runner) {
            this.runner = runner;
            this.executor.input(runner.pipelineRunner().getDelegate());
        }

        private void request(long demand) {
            runner.pipelineRunner().getDelegate().request(demand);
        }

        @Override
        public void close() {
            runner.completeGracefully();
        }
    }
}
