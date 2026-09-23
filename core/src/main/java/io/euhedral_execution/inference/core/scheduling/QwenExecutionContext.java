package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;

/// Mutable state for one inference quantum. Frames own operations; this object owns their buffers
/// and completion accounting, while its sequence reference retains sequence-lifetime state.
public final class QwenExecutionContext {

    private static final Throwable TERMINAL_SUCCESS = new IllegalStateException("quantum already finalized");
    private static final Runnable NO_OP = () -> {};

    public enum ExecutionKind {
        PREFILL,
        DECODE
    }

    public enum Status {
        SUCCESS,
        CANCELLED,
        FAILED
    }

    public record Outcome(Status status, Throwable failure) {}

    private final QwenExecutionPlan plan;
    private final QwenSequenceState sequence;
    private final ExecutionKind kind;
    private final long startPosition;
    private final int[] tokenIds;
    private final AtomicBoolean submitted = new AtomicBoolean();
    private final AtomicInteger outstanding = new AtomicInteger();
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final CompletableFuture<Outcome> outcome = new CompletableFuture<>();
    private final AtomicIntegerArray remainingDependencies;
    private QwenSequenceState.ExecutionLease lease;
    private QwenExecutionWorkspace workspace;
    private long temporaryTokenIdsAddress;

    public QwenExecutionContext(
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            ExecutionKind kind,
            long startPosition,
            int[] tokenIds) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.kind = Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(tokenIds, "tokenIds");
        if (startPosition < 0 || tokenIds.length == 0) {
            throw new IllegalArgumentException("invalid token range");
        }
        this.startPosition = startPosition;
        this.tokenIds = tokenIds.clone();
        this.remainingDependencies = new AtomicIntegerArray(plan.instructions().size());
        for (QwenExecutionPlan.Instruction instruction : plan.instructions()) {
            remainingDependencies.set(
                    instruction.id(), instruction.dependencies().size());
        }
    }

    public QwenExecutionPlan plan() {
        return this.plan;
    }

    public QwenSequenceState sequenceState() {
        return this.sequence;
    }

    public int inputTokenCount() {
        return this.tokenIds.length;
    }

    public int[] inputTokenIds() {
        return this.tokenIds.clone();
    }

    public QwenExecutionWorkspace workspace() {
        QwenExecutionWorkspace current = this.workspace;
        if (current == null) {
            throw new IllegalStateException("workspace has not been allocated");
        }
        return current;
    }

    public CompletableFuture<Outcome> outcome() {
        return this.outcome.copy();
    }

    CompletableFuture<Outcome> completion() {
        return this.outcome;
    }

    public void cancel() {
        this.sequence.cancel();
    }

    public void fail(Throwable cause) {
        Objects.requireNonNull(cause, "cause");
        if (this.failure.compareAndSet(null, cause)) return;
        Throwable first = this.failure.get();
        if (first != TERMINAL_SUCCESS && first != cause) first.addSuppressed(cause);
    }

    boolean hasFailureOrCancellation() {
        return this.failure.get() != null || this.sequence.cancellationRequested();
    }

    Throwable failure() {
        return this.failure.get();
    }

    boolean dependencyCompleted(int instructionId) {
        int left = this.remainingDependencies.decrementAndGet(instructionId);
        if (left < 0) {
            throw new IllegalStateException("instruction dependency completed twice");
        }
        return left == 0;
    }

    void reserveWork() {
        this.outstanding.incrementAndGet();
    }

    boolean releaseWork() {
        int left = this.outstanding.decrementAndGet();
        if (left < 0) {
            throw new IllegalStateException("work completion exceeded admission");
        }
        return left == 0;
    }

    long allocateTemporaryTokenIds(QwenExecutionGpu gpu, long bytes) {
        this.temporaryTokenIdsAddress = gpu.allocate(bytes);
        if (this.temporaryTokenIdsAddress == 0) {
            throw new IllegalStateException("GPU returned a null token-ID address");
        }
        return this.temporaryTokenIdsAddress;
    }

    void releaseTemporaryTokenIds(QwenExecutionGpu gpu) {
        if (this.temporaryTokenIdsAddress != 0) {
            gpu.free(this.temporaryTokenIdsAddress);
            this.temporaryTokenIdsAddress = 0;
        }
    }

    void begin(QwenExecutionGpu gpu) {
        begin(gpu, NO_OP);
    }

    /// Package-private hook to deterministically exercise cancellation at the lease-claim boundary.
    void begin(QwenExecutionGpu gpu, Runnable beforeClaim) {
        if (!this.submitted.compareAndSet(false, true)) {
            throw new IllegalStateException("quantum was already submitted");
        }
        if (this.sequence.cancellationRequested()) {
            this.outcome.complete(new Outcome(Status.CANCELLED, null));
            return;
        }
        try {
            long end = Math.addExact(this.startPosition, this.tokenIds.length);
            if (end < 0) {
                throw new IllegalArgumentException("token range overflows");
            }
            int vocabulary = this.plan.weights().config().vocabSize();
            for (int index = 0; index < this.tokenIds.length; index++) {
                if (this.tokenIds[index] < 0 || this.tokenIds[index] >= vocabulary) {
                    throw new IllegalArgumentException("token ID outside vocabulary at index " + index);
                }
            }
            beforeClaim.run();
            try {
                this.lease = this.sequence.claimExecution(this.startPosition);
            } catch (IllegalStateException claimFailure) {
                if (this.sequence.terminalState() == QwenSequenceState.TerminalState.CANCELLED) {
                    this.outcome.complete(new Outcome(Status.CANCELLED, null));
                    return;
                }
                throw claimFailure;
            }
            this.workspace = new QwenExecutionWorkspace(
                    gpu, this.tokenIds.length, this.plan.weights().config().hiddenSize(), this.plan.projectionWidths());
            this.workspace.allocateBuffers();
        } catch (RuntimeException | Error error) {
            fail(error);
            finish(null, gpu);
        }
    }

    /// Runs only after all admitted frames have finished and no frame can still access the buffers.
    void finish(java.util.function.Consumer<? super QwenExecutionContext> terminalConsumer, QwenExecutionGpu gpu) {
        if (this.outcome.isDone()) {
            return;
        }
        try {
            releaseTemporaryTokenIds(gpu);
        } catch (Throwable cleanupFailure) {
            fail(cleanupFailure);
        }
        if (this.failure.get() == null && !this.sequence.cancellationRequested() && terminalConsumer != null) {
            try {
                terminalConsumer.accept(this);
            } catch (Throwable consumerFailure) {
                fail(consumerFailure);
            }
        }
        if (this.workspace != null) {
            try {
                this.workspace.close();
            } catch (Throwable cleanupFailure) {
                fail(cleanupFailure);
                try {
                    this.workspace.close();
                } catch (Throwable retryFailure) {
                    fail(retryFailure);
                }
            }
        }
        Throwable error = this.failure.get();
        if (error == null && !this.failure.compareAndSet(null, TERMINAL_SUCCESS)) {
            error = this.failure.get();
        }
        Outcome completed;
        if (error != null) {
            if (this.lease != null) {
                this.sequence.markFailed(this.lease, error);
            }
            completed = new Outcome(Status.FAILED, error);
        } else if (this.sequence.cancellationRequested()) {
            if (this.lease != null) {
                this.sequence.markCancelled(this.lease);
            }
            completed = new Outcome(Status.CANCELLED, null);
        } else {
            boolean cancelled = this.sequence.releaseExecutionAndCheckCancellation(
                    this.lease, this.startPosition + this.tokenIds.length);
            completed = new Outcome(cancelled ? Status.CANCELLED : Status.SUCCESS, null);
        }
        this.lease = null;
        this.outcome.complete(completed);
    }
}
