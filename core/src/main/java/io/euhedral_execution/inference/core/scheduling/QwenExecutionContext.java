package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.frames.PipelineFrame;
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
    private final int inputTokenCount;
    private final List<String> executionTrace = new ArrayList<>();
    private Throwable requestedFailure;
    private Throwable pendingFailure;
    private boolean pendingCancellation;
    private QwenSequenceState.ExecutionLease executionLease;
    private Result result;

    public QwenExecutionContext(
            QwenExecutionPlan plan,
            QwenSequenceState sequenceState,
            ExecutionKind executionKind,
            long inputStartPosition,
            int inputTokenCount) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequenceState = Objects.requireNonNull(sequenceState, "sequenceState");
        this.executionKind = Objects.requireNonNull(executionKind, "executionKind");
        if (inputStartPosition < 0) {
            throw new IllegalArgumentException("inputStartPosition must be non-negative");
        }
        if (inputTokenCount <= 0) {
            throw new IllegalArgumentException("inputTokenCount must be positive");
        }
        this.inputStartPosition = inputStartPosition;
        this.inputTokenCount = inputTokenCount;
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
        return this;
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
        if (observerFailure != null) {
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
