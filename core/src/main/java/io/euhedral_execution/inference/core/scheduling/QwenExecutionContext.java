package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.frames.PipelineFrame;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Per-submission state passed through the reusable Euhedral pipeline.
///
/// This context references model-plan and sequence state; it does not own model weights.
public final class QwenExecutionContext {

    public enum ExecutionKind {
        PREFILL,
        DECODE
    }

    public record Result(
            ExecutionKind executionKind,
            long sequenceId,
            long inputStartPosition,
            long nextTokenPosition,
            List<String> trace) {}

    private final QwenExecutionPlan plan;
    private final QwenSequenceState sequenceState;
    private final ExecutionKind executionKind;
    private final long inputStartPosition;
    private final int[] inputTokenIds;
    private final int inputTokenCount;
    private final List<String> executionTrace = new ArrayList<>();
    private Throwable requestedFailure;
    private Throwable pendingFailure;
    private boolean pendingCancellation;
    private QwenSequenceState.ExecutionLease executionLease;
    private QwenExecutionWorkspace workspace;
    private long temporaryTokenIdsAddress;
    private Result result;

    public QwenExecutionContext(
            QwenExecutionPlan plan,
            QwenSequenceState sequenceState,
            ExecutionKind executionKind,
            long inputStartPosition,
            int[] inputTokenIds) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequenceState = Objects.requireNonNull(sequenceState, "sequenceState");
        this.executionKind = Objects.requireNonNull(executionKind, "executionKind");
        Objects.requireNonNull(inputTokenIds, "inputTokenIds");
        if (inputStartPosition < 0) {
            throw new IllegalArgumentException("inputStartPosition must be non-negative");
        }
        if (inputTokenIds.length <= 0) {
            throw new IllegalArgumentException("inputTokenIds must not be empty");
        }
        this.inputStartPosition = inputStartPosition;
        this.inputTokenIds = inputTokenIds.clone();
        this.inputTokenCount = inputTokenIds.length;
    }

    public QwenExecutionPlan plan() {
        return this.plan;
    }

    public QwenSequenceState sequenceState() {
        return this.sequenceState;
    }

    public ExecutionKind executionKind() {
        return this.executionKind;
    }

    public long inputStartPosition() {
        return this.inputStartPosition;
    }

    public int inputTokenCount() {
        return this.inputTokenCount;
    }

    /// Returns a copy of the token IDs submitted with this execution.
    public int[] inputTokenIds() {
        return this.inputTokenIds.clone();
    }

    /// Returns the per-submission hidden-state workspace after embedding has started.
    public QwenExecutionWorkspace workspace() {
        if (this.workspace == null) {
            throw new IllegalStateException("Qwen execution workspace has not been allocated");
        }
        return this.workspace;
    }

    public List<String> executionTrace() {
        return List.copyOf(this.executionTrace);
    }

    public Result result() {
        return this.result;
    }

    public boolean isTerminal() {
        return this.result != null;
    }

    /// Requests cancellation before or during this context's pipeline execution.
    public void cancel() {
        this.sequenceState.cancel();
    }

    /// Requests a deterministic stage failure for failure-path validation.
    public void fail(Throwable failure) {
        this.requestedFailure = Objects.requireNonNull(failure, "failure");
    }

    QwenExecutionContext prepare(QwenExecutionPlan expectedPlan) {
        if (this.plan != expectedPlan) {
            throw new IllegalArgumentException("Execution context belongs to a different Qwen execution plan");
        }
        beginExecution();
        this.executionTrace.add("prepare/embed");
        try {
            checkActive();
            TensorHandle embedding = validateEmbedding();
            QwenConfig config = this.plan.weights().config();
            this.workspace = new QwenExecutionWorkspace(this.plan.gpu(), this.inputTokenCount, config.hiddenSize());
            uploadAndEmbed(embedding, config);
        } catch (RuntimeException | Error failure) {
            Throwable cleanupFailure = closeSubmissionResources();
            if (cleanupFailure != null) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
        return this;
    }

    private TensorHandle validateEmbedding() {
        QwenWeights weights = this.plan.weights();
        QwenConfig config = weights.config();
        if (config == null || config.vocabSize() <= 0 || config.hiddenSize() <= 0) {
            throw new IllegalStateException("Qwen embedding configuration is missing or invalid");
        }
        TensorHandle embedding = weights.tokenEmbedding();
        if (embedding == null) {
            throw new IllegalStateException("Qwen token embedding object is missing");
        }
        if (embedding.deviceAddress() == 0) {
            throw new IllegalStateException("Qwen token embedding has no resident GPU allocation");
        }
        if (embedding.dataType() != TensorDataType.BF16
                || embedding.format() != WeightFormat.Q3_G64_FP16
                || embedding.layout() != WeightLayout.ROW_SPLIT_K128_V1) {
            throw new IllegalArgumentException(
                    "Qwen token embedding format/layout mismatch: expected BF16/Q3_G64_FP16/ROW_SPLIT_K128_V1");
        }
        long[] shape = embedding.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != config.vocabSize()
                || shape[1] != config.hiddenSize()
                || shape[1] % 64 != 0) {
            throw new IllegalArgumentException("Qwen token embedding shape does not match its configuration");
        }
        if (!weights.runtimeObjects().isEmpty()) {
            TensorHandle loadedEmbedding = weights.runtimeObjects().get(embedding.name());
            if (loadedEmbedding == null || loadedEmbedding.deviceAddress() != embedding.deviceAddress()) {
                throw new IllegalStateException(
                        "loaded Qwen token embedding object is missing from the runtime inventory");
            }
        }
        long expectedByteSize;
        try {
            expectedByteSize = CompactTensorLayout.expectedByteSize(
                    shape, embedding.dataType(), embedding.format(), embedding.layout());
        } catch (IllegalArgumentException mismatch) {
            throw new IllegalArgumentException("Qwen token embedding format/layout metadata is invalid", mismatch);
        }
        if (embedding.byteSize() != expectedByteSize) {
            throw new IllegalArgumentException("Qwen token embedding byte size does not match its format/layout");
        }
        for (int index = 0; index < this.inputTokenIds.length; index++) {
            int tokenId = this.inputTokenIds[index];
            if (tokenId < 0 || tokenId >= config.vocabSize()) {
                throw new IllegalArgumentException("input token ID at index " + index + " is outside the vocabulary");
            }
        }
        return embedding;
    }

    private void uploadAndEmbed(TensorHandle embedding, QwenConfig config) {
        QwenExecutionGpu gpu = this.plan.gpu();
        long tokenBytes = Math.multiplyExact((long) this.inputTokenCount, Integer.BYTES);
        Throwable executionFailure = null;
        try (Arena inputArena = Arena.ofConfined()) {
            MemorySegment hostTokenIds = inputArena.allocate(tokenBytes, Integer.BYTES);
            for (int index = 0; index < this.inputTokenCount; index++) {
                hostTokenIds.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, this.inputTokenIds[index]);
            }
            if (this.temporaryTokenIdsAddress != 0) {
                throw new IllegalStateException("temporary token-ID buffer is already allocated");
            }
            this.temporaryTokenIdsAddress = gpu.allocate(tokenBytes);
            if (this.temporaryTokenIdsAddress == 0) {
                throw new IllegalStateException("GPU returned a null token-ID buffer address");
            }
            gpu.copyHostToDevice(this.temporaryTokenIdsAddress, hostTokenIds, tokenBytes);
            gpu.embedQ3(
                    this.temporaryTokenIdsAddress,
                    embedding.deviceAddress(),
                    embedding.byteSize(),
                    this.workspace.hiddenStateAddress(),
                    this.inputTokenCount,
                    config.vocabSize(),
                    config.hiddenSize());
            gpu.synchronize();
            checkActive();
        } catch (RuntimeException | Error failure) {
            executionFailure = failure;
            throw failure;
        } finally {
            if (this.temporaryTokenIdsAddress != 0) {
                try {
                    releaseTemporaryTokenIds();
                } catch (RuntimeException | Error cleanupFailure) {
                    if (executionFailure == null) {
                        throw cleanupFailure;
                    }
                    executionFailure.addSuppressed(cleanupFailure);
                }
            }
        }
    }

    QwenExecutionContext executeLayer(QwenExecutionPlan.LayerOperation operation) {
        checkActive();
        this.executionTrace.add("layer[" + operation.index() + "]:" + operation.type());
        return this;
    }

    QwenExecutionContext executeFinal() {
        checkActive();
        this.executionTrace.add("final-norm/lm-head");
        return this;
    }

    void captureTerminal() {
        checkActive();
        this.executionTrace.add("terminal-result");
        long nextTokenPosition = this.inputStartPosition + this.inputTokenCount;
        if (nextTokenPosition < this.inputStartPosition) {
            abortFailure(new IllegalArgumentException("input token range overflows"));
            throw new IllegalArgumentException("input token range overflows");
        }
        this.result = new Result(
                this.executionKind,
                this.sequenceState.sequenceId(),
                this.inputStartPosition,
                nextTokenPosition,
                List.copyOf(this.executionTrace));
    }

    void checkTerminalBoundary() {
        checkActive();
    }

    PipelineFrame.Outcome completeOutcome(PipelineFrame.Outcome outcome, Throwable observerFailure) {
        Throwable submissionCleanupFailure = closeSubmissionResources();
        if (observerFailure != null) {
            if (submissionCleanupFailure != null) {
                observerFailure.addSuppressed(submissionCleanupFailure);
            }
            this.result = null;
            if (this.executionLease != null) {
                this.sequenceState.markFailed(this.executionLease, observerFailure);
                this.executionLease = null;
            } else if (this.sequenceState.terminalState() == QwenSequenceState.TerminalState.ACTIVE) {
                this.sequenceState.markFailedBeforeClaim(observerFailure);
            }
            this.pendingFailure = null;
            return new PipelineFrame.Outcome(PipelineFrame.Status.FAILED, observerFailure);
        }
        Objects.requireNonNull(outcome, "outcome");
        if (submissionCleanupFailure != null) {
            Throwable failure = outcome.failure();
            if (failure == null) {
                failure = submissionCleanupFailure;
            } else {
                failure.addSuppressed(submissionCleanupFailure);
            }
            outcome = new PipelineFrame.Outcome(PipelineFrame.Status.FAILED, failure);
        }
        switch (outcome.status()) {
            case SUCCESS -> {
                if (this.executionLease == null) {
                    this.result = null;
                    return new PipelineFrame.Outcome(
                            PipelineFrame.Status.FAILED,
                            new IllegalStateException("Successful Qwen outcome has no active execution lease"));
                } else if (this.pendingCancellation) {
                    this.sequenceState.markCancelled(this.executionLease);
                    this.executionLease = null;
                    this.result = null;
                    this.pendingCancellation = false;
                    return new PipelineFrame.Outcome(PipelineFrame.Status.CANCELLED, null);
                } else {
                    boolean cancellationWon = this.sequenceState.releaseExecutionAndCheckCancellation(
                            this.executionLease, this.result.nextTokenPosition());
                    this.executionLease = null;
                    if (cancellationWon) {
                        this.result = null;
                        return new PipelineFrame.Outcome(PipelineFrame.Status.CANCELLED, null);
                    }
                }
                this.pendingCancellation = false;
                return outcome;
            }
            case CANCELLED -> {
                this.result = null;
                if (this.executionLease != null) {
                    this.sequenceState.markCancelled(this.executionLease);
                    this.executionLease = null;
                }
                this.pendingCancellation = false;
                return outcome;
            }
            case FAILED -> {
                this.result = null;
                Throwable failure = this.pendingFailure != null
                        ? this.pendingFailure
                        : outcome.failure() == null
                                ? new IllegalStateException("Qwen pipeline failed without a cause")
                                : outcome.failure();
                if (this.executionLease != null) {
                    this.sequenceState.markFailed(this.executionLease, failure);
                    this.executionLease = null;
                }
                this.pendingFailure = null;
                return outcome;
            }
            case FILTERED -> {
                this.result = null;
                if (this.executionLease != null) {
                    this.sequenceState.markFailed(
                            this.executionLease, new IllegalStateException("Qwen execution was filtered"));
                    this.executionLease = null;
                }
                this.pendingFailure = null;
                return outcome;
            }
        }
        throw new IllegalStateException("Unhandled Qwen pipeline outcome: " + outcome.status());
    }

    private void beginExecution() {
        if (this.sequenceState.cancellationRequested()) {
            abortCancellation();
            throw AbstractFrame.CANCEL_SIGNAL;
        }
        if (this.requestedFailure != null) {
            abortFailure(this.requestedFailure);
            throwFailure(this.requestedFailure);
        }
        try {
            this.executionLease = this.sequenceState.claimExecution(this.inputStartPosition);
        } catch (IllegalStateException failure) {
            if (this.sequenceState.cancellationRequested()) {
                abortCancellation();
                throw AbstractFrame.CANCEL_SIGNAL;
            }
            throw failure;
        }
    }

    private void checkActive() {
        if (this.sequenceState.cancellationRequested()) {
            abortCancellation();
            throw AbstractFrame.CANCEL_SIGNAL;
        }
        if (this.requestedFailure != null) {
            abortFailure(this.requestedFailure);
            throwFailure(this.requestedFailure);
        }
    }

    private void abortCancellation() {
        if (this.executionLease != null) {
            this.pendingCancellation = true;
        } else if (this.sequenceState.terminalState() == QwenSequenceState.TerminalState.ACTIVE) {
            this.sequenceState.markCancelledBeforeClaim();
        }
    }

    private void abortFailure(Throwable failure) {
        if (this.executionLease != null) {
            this.pendingFailure = failure;
        } else if (this.sequenceState.terminalState() == QwenSequenceState.TerminalState.ACTIVE) {
            this.sequenceState.markFailedBeforeClaim(failure);
        }
    }

    void invalidateAfterTerminalConsumerFailure(Throwable failure) {
        this.result = null;
        if (this.executionLease != null) {
            this.pendingFailure = failure;
        } else {
            this.sequenceState.markFailedAfterRelease(failure);
        }
    }

    private Throwable closeSubmissionResources() {
        Throwable cleanupFailure = null;
        try {
            releaseTemporaryTokenIds();
        } catch (Throwable failure) {
            cleanupFailure = failure;
        }
        if (this.workspace != null && !this.workspace.isClosed()) {
            try {
                this.workspace.close();
            } catch (Throwable failure) {
                if (cleanupFailure == null) {
                    cleanupFailure = failure;
                } else if (cleanupFailure != failure) {
                    cleanupFailure.addSuppressed(failure);
                }
            }
        }
        return cleanupFailure;
    }

    private void releaseTemporaryTokenIds() {
        if (this.temporaryTokenIdsAddress == 0) {
            return;
        }
        this.plan.gpu().free(this.temporaryTokenIdsAddress);
        this.temporaryTokenIdsAddress = 0;
    }

    private static void throwFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) {
            throw runtimeException;
        }
        throw new QwenExecutionFailure(failure);
    }

    private static final class QwenExecutionFailure extends RuntimeException {

        private QwenExecutionFailure(Throwable cause) {
            super(cause);
        }
    }
}
