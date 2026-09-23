package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.PipelineFrame;
import io.euhedral_execution.core.generics.AbstractExecutor;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.core.ingest.PipelineRunner;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class QwenExecutionContextTest {

    @Test
    void prepareUploadsIdsEmbedsSynchronouslyAndReleasesOnlySubmissionBuffers() throws Exception {
        QwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        QwenWeights weights = weights();
        QwenExecutionPlan plan = new QwenExecutionPlan(weights, gpu);
        QwenSequenceState sequence = new QwenSequenceState(91L);
        int[] tokens = {3, 1, 7};
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0L, tokens);
        long[] outputAddress = {0};
        long[] outputBytes = {0};
        QwenExecutionRunner runner = plan.newRunner(terminalContext -> {
            QwenExecutionWorkspace workspace = terminalContext.workspace();
            assertFalse(workspace.isClosed());
            outputAddress[0] = workspace.hiddenStateAddress();
            outputBytes[0] = workspace.byteSize();
        });

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);

            assertEquals(
                    PipelineFrame.Status.SUCCESS,
                    outcome.get(5, TimeUnit.SECONDS).status());
        }

        RecordingQwenExecutionGpu recordingGpu = (RecordingQwenExecutionGpu) gpu;
        assertNotNull(context.workspace());
        assertTrue(context.workspace().isClosed());
        assertEquals(3L * 64 * Short.BYTES, outputBytes[0]);
        assertEquals(List.of(outputBytes[0], (long) tokens.length * Integer.BYTES), recordingGpu.allocationSizes());
        assertEquals(List.of(3, 1, 7), recordingGpu.uploadedTokenIds());
        assertEquals(outputAddress[0], recordingGpu.outputAddress());
        assertEquals(weights.tokenEmbedding().deviceAddress(), recordingGpu.embeddingAddress());
        assertEquals(1, recordingGpu.synchronizations());
        assertEquals(Set.copyOf(recordingGpu.allocations()), Set.copyOf(recordingGpu.frees()));
        assertFalse(recordingGpu.frees().contains(weights.tokenEmbedding().deviceAddress()));
        assertTrue(recordingGpu.events().indexOf("synchronize")
                < recordingGpu.events().indexOf("free-token-ids"));
    }

    @Test
    void embeddingFailureReleasesTokenIdsAndWorkspaceWithoutFreeingWeights() throws Exception {
        IllegalStateException failure = new IllegalStateException("injected embedding failure");
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        gpu.embeddingFailure = failure;
        QwenWeights weights = weights();
        QwenExecutionPlan plan = new QwenExecutionPlan(weights, gpu);
        QwenSequenceState sequence = new QwenSequenceState(92L);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3, 1, 7});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);

            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertSame(failure, completed.failure());
        }

        assertTrue(context.workspace().isClosed());
        assertEquals(Set.copyOf(gpu.allocations()), Set.copyOf(gpu.frees()));
        assertFalse(gpu.frees().contains(weights.tokenEmbedding().deviceAddress()));
    }

    @Test
    void cancellationDuringSynchronousEmbeddingReleasesBothSubmissionBuffers() throws Exception {
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        QwenWeights weights = weights();
        QwenExecutionPlan plan = new QwenExecutionPlan(weights, gpu);
        QwenSequenceState sequence = new QwenSequenceState(93L);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3, 1, 7});
        gpu.afterEmbedding = context::cancel;
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);

            assertEquals(
                    PipelineFrame.Status.CANCELLED,
                    outcome.get(5, TimeUnit.SECONDS).status());
        }

        assertEquals(QwenSequenceState.TerminalState.CANCELLED, sequence.terminalState());
        assertTrue(context.workspace().isClosed());
        assertEquals(Set.copyOf(gpu.allocations()), Set.copyOf(gpu.frees()));
        assertFalse(gpu.frees().contains(weights.tokenEmbedding().deviceAddress()));
    }

    @Test
    void rejectsOutOfVocabularyTokenBeforeAllocatingSubmissionBuffers() throws Exception {
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(), gpu);
        QwenSequenceState sequence = new QwenSequenceState(94L);
        QwenExecutionContext context =
                new QwenExecutionContext(plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {8});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);

            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertTrue(completed.failure().getMessage().contains("outside the vocabulary"));
        }

        assertTrue(gpu.allocations().isEmpty());
    }

    @Test
    void tokenIdAllocationFailureReleasesWorkspace() throws Exception {
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        gpu.failAllocationNumber = 2;
        IllegalStateException failure = new IllegalStateException("injected allocation failure");
        gpu.allocationFailure = failure;
        QwenExecutionPlan plan = new QwenExecutionPlan(weights(), gpu);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, new QwenSequenceState(95L), QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3, 1, 7});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);
            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertSame(failure, completed.failure());
        }

        assertTrue(context.workspace().isClosed());
        assertEquals(gpu.allocations(), gpu.frees());
    }

    @Test
    void tokenIdUploadFailureReleasesTokenBufferAndWorkspace() throws Exception {
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        IllegalStateException failure = new IllegalStateException("injected upload failure");
        gpu.uploadFailure = failure;
        QwenWeights weights = weights();
        QwenExecutionPlan plan = new QwenExecutionPlan(weights, gpu);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, new QwenSequenceState(96L), QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3, 1, 7});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);
            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertSame(failure, completed.failure());
        }

        assertTrue(context.workspace().isClosed());
        assertEquals(Set.copyOf(gpu.allocations()), Set.copyOf(gpu.frees()));
        assertFalse(gpu.frees().contains(weights.tokenEmbedding().deviceAddress()));
    }

    @Test
    void tokenIdFreeFailureRetainsAddressForTerminalCleanupRetry() throws Exception {
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        gpu.freeFailuresRemaining = 1;
        QwenWeights weights = weights();
        QwenExecutionPlan plan = new QwenExecutionPlan(weights, gpu);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, new QwenSequenceState(99L), QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3, 1, 7});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);
            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertTrue(completed.failure().getMessage().contains("synthetic device free failure"));
        }

        assertTrue(context.workspace().isClosed());
        assertEquals(Set.copyOf(gpu.allocations()), Set.copyOf(gpu.frees()));
        assertEquals(3, gpu.freeAttempts());
        assertFalse(gpu.frees().contains(weights.tokenEmbedding().deviceAddress()));
    }

    @Test
    void missingEmbeddingFailsBeforeWorkspaceAllocation() throws Exception {
        QwenWeights complete = weights();
        QwenWeights missing = new QwenWeights(
                complete.config(), null, complete.layers(), complete.finalNorm(), complete.lmHead(), complete.mtp());
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        QwenExecutionPlan plan = new QwenExecutionPlan(missing, gpu);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, new QwenSequenceState(97L), QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);
            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertTrue(completed.failure().getMessage().contains("token embedding object is missing"));
        }

        assertTrue(gpu.allocations().isEmpty());
    }

    @Test
    void embeddingFormatOrLayoutMismatchFailsBeforeWorkspaceAllocation() throws Exception {
        QwenWeights complete = weights();
        TensorHandle incompatibleEmbedding = new TensorHandle(
                "text/token_embedding",
                new long[] {8, 64},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                77L,
                8L * 64 * Short.BYTES);
        QwenWeights incompatible = new QwenWeights(
                complete.config(),
                incompatibleEmbedding,
                complete.layers(),
                complete.finalNorm(),
                complete.lmHead(),
                complete.mtp());
        RecordingQwenExecutionGpu gpu = new RecordingQwenExecutionGpu();
        QwenExecutionPlan plan = new QwenExecutionPlan(incompatible, gpu);
        QwenExecutionContext context = new QwenExecutionContext(
                plan, new QwenSequenceState(98L), QwenExecutionContext.ExecutionKind.PREFILL, 0L, new int[] {3});
        QwenExecutionRunner runner = plan.newRunner();

        try (RunnerDriver driver = new RunnerDriver(runner)) {
            CompletableFuture<PipelineFrame.Outcome> outcome = runner.submit(context);
            driver.request(8);
            PipelineFrame.Outcome completed = outcome.get(5, TimeUnit.SECONDS);
            assertEquals(PipelineFrame.Status.FAILED, completed.status());
            assertTrue(completed.failure().getMessage().contains("format/layout mismatch"));
        }

        assertTrue(gpu.allocations().isEmpty());
    }

    private static QwenWeights weights() {
        int vocabulary = 8;
        int hiddenSize = 64;
        long embeddingBytes = CompactTensorLayout.expectedByteSize(
                new long[] {vocabulary, hiddenSize},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1);
        TensorHandle embedding = new TensorHandle(
                "text/token_embedding",
                new long[] {vocabulary, hiddenSize},
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                77L,
                embeddingBytes);
        QwenConfig config = new QwenConfig(
                vocabulary,
                hiddenSize,
                0,
                1,
                1,
                hiddenSize,
                hiddenSize,
                1,
                1,
                1,
                1,
                1,
                1.0e-6,
                1_000_000.0,
                1.0,
                128,
                "silu",
                new QwenLayerType[0],
                0,
                0,
                0,
                0,
                false,
                false,
                0);
        return new QwenWeights(config, embedding, new QwenLayerWeights[0], embedding, embedding, null);
    }

    private static final class RecordingQwenExecutionGpu implements QwenExecutionGpu {

        private final List<Long> allocations = new ArrayList<>();
        private final List<Long> frees = new ArrayList<>();
        private final List<Long> allocationSizes = new ArrayList<>();
        private final Map<Long, int[]> uploadedTokens = new HashMap<>();
        private final List<String> events = new ArrayList<>();
        private long nextAddress = 1_000;
        private long outputAddress;
        private long embeddingAddress;
        private int synchronizations;
        private int freeFailuresRemaining;
        private int freeAttempts;
        private RuntimeException embeddingFailure;
        private Runnable afterEmbedding;
        private int failAllocationNumber = -1;
        private RuntimeException allocationFailure;
        private RuntimeException uploadFailure;

        @Override
        public long allocate(long byteSize) {
            if (this.allocations.size() + 1 == this.failAllocationNumber) {
                throw this.allocationFailure;
            }
            long address = this.nextAddress++;
            this.allocations.add(address);
            this.allocationSizes.add(byteSize);
            this.events.add("allocate-" + byteSize);
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            if (this.uploadFailure != null) {
                throw this.uploadFailure;
            }
            int[] tokenIds = new int[Math.toIntExact(byteSize / Integer.BYTES)];
            for (int index = 0; index < tokenIds.length; index++) {
                tokenIds[index] = source.get(ValueLayout.JAVA_INT, (long) index * Integer.BYTES);
            }
            this.uploadedTokens.put(destination, tokenIds);
            this.events.add("upload-token-ids");
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void free(long address) {
            this.freeAttempts++;
            if (this.freeFailuresRemaining > 0) {
                this.freeFailuresRemaining--;
                throw new IllegalStateException("synthetic device free failure");
            }
            this.frees.add(address);
            this.events.add(this.uploadedTokens.containsKey(address) ? "free-token-ids" : "free-workspace");
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            assertEquals(
                    List.of(3, 1, 7),
                    Arrays.stream(this.uploadedTokens.get(tokenIdsAddress))
                            .boxed()
                            .toList());
            assertEquals(3, tokenCount);
            assertEquals(8, vocabularySize);
            assertEquals(64, hiddenSize);
            assertTrue(embeddingByteSize > 0);
            this.embeddingAddress = embeddingAddress;
            this.outputAddress = hiddenStateAddress;
            this.events.add("embed-q3");
            if (this.afterEmbedding != null) {
                this.afterEmbedding.run();
            }
            if (this.embeddingFailure != null) {
                throw this.embeddingFailure;
            }
        }

        @Override
        public void synchronize() {
            this.synchronizations++;
            this.events.add("synchronize");
        }

        private List<Long> allocations() {
            return List.copyOf(this.allocations);
        }

        private List<Long> frees() {
            return List.copyOf(this.frees);
        }

        private int freeAttempts() {
            return this.freeAttempts;
        }

        private List<Long> allocationSizes() {
            return List.copyOf(this.allocationSizes);
        }

        private List<Integer> uploadedTokenIds() {
            return Arrays.stream(this.uploadedTokens.values().iterator().next())
                    .boxed()
                    .toList();
        }

        private List<String> events() {
            return List.copyOf(this.events);
        }

        private long outputAddress() {
            return this.outputAddress;
        }

        private long embeddingAddress() {
            return this.embeddingAddress;
        }

        private int synchronizations() {
            return this.synchronizations;
        }
    }

    private static final class RunnerDriver implements AutoCloseable {

        private final QwenExecutionRunner runner;
        private final AbstractExecutor executor = new DefaultExecutor();

        private RunnerDriver(QwenExecutionRunner runner) {
            this.runner = runner;
            PipelineRunner<QwenExecutionContext> pipeline = runner.pipelineRunner();
            this.executor.input(pipeline.getDelegate());
        }

        private void request(long demand) {
            this.runner.pipelineRunner().getDelegate().request(demand);
        }

        @Override
        public void close() {
            this.runner.completeGracefully();
        }
    }
}
