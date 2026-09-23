package io.euhedral_execution.inference.core.scheduling;

import java.util.Objects;

/// Request-lifetime state shared by successive Qwen execution quanta.
///
/// The execution lease is intentionally exclusive: one submitted chain may mutate this state at a
/// time. KV-cache and recurrent state are opaque placeholders until their ownership contracts exist.
public final class QwenSequenceState {

    public enum TerminalState {
        ACTIVE,
        CANCELLED,
        FAILED,
        COMPLETED
    }

    /// Capability held by the one chain currently allowed to mutate sequence state.
    public static final class ExecutionLease {

        private final QwenSequenceState owner;
        private boolean active = true;

        private ExecutionLease(QwenSequenceState owner) {
            this.owner = owner;
        }
    }

    private final long sequenceId;
    private long currentTokenPosition;
    private boolean executionClaimed;
    private ExecutionLease activeLease;
    private boolean cancellationRequested;
    private TerminalState terminalState = TerminalState.ACTIVE;
    private Throwable terminalFailure;
    private Object kvCacheState;
    private Object recurrentState;

    public QwenSequenceState(long sequenceId) {
        this(sequenceId, 0L);
    }

    public QwenSequenceState(long sequenceId, long initialTokenPosition) {
        if (sequenceId < 0) {
            throw new IllegalArgumentException("sequenceId must be non-negative");
        }
        if (initialTokenPosition < 0) {
            throw new IllegalArgumentException("initialTokenPosition must be non-negative");
        }
        this.sequenceId = sequenceId;
        this.currentTokenPosition = initialTokenPosition;
    }

    public long sequenceId() {
        return this.sequenceId;
    }

    public synchronized long currentTokenPosition() {
        return this.currentTokenPosition;
    }

    public synchronized boolean isExecutionClaimed() {
        return this.executionClaimed;
    }

    public synchronized boolean cancellationRequested() {
        return this.cancellationRequested;
    }

    public synchronized TerminalState terminalState() {
        return this.terminalState;
    }

    public synchronized Throwable terminalFailure() {
        return this.terminalFailure;
    }

    public synchronized Object kvCacheState() {
        return this.kvCacheState;
    }

    public synchronized void setKvCacheState(ExecutionLease lease, Object kvCacheState) {
        requireLease(lease);
        this.kvCacheState = kvCacheState;
    }

    public synchronized Object recurrentState() {
        return this.recurrentState;
    }

    public synchronized void setRecurrentState(ExecutionLease lease, Object recurrentState) {
        requireLease(lease);
        this.recurrentState = recurrentState;
    }

    /// Requests cancellation of the active or next execution quantum.
    public synchronized void cancel() {
        if (this.terminalState == TerminalState.ACTIVE) {
            this.cancellationRequested = true;
            if (!this.executionClaimed) {
                this.terminalState = TerminalState.CANCELLED;
            }
        }
    }

    /// Marks the sequence terminal after its owner has stopped admitting work.
    public synchronized void complete() {
        if (this.executionClaimed) {
            throw new IllegalStateException("Cannot complete a sequence during execution");
        }
        if (this.terminalState != TerminalState.ACTIVE) {
            return;
        }
        this.terminalState = TerminalState.COMPLETED;
    }

    /// Claims exclusive mutation for one execution quantum.
    public synchronized ExecutionLease claimExecution(long expectedStartPosition) {
        if (this.executionClaimed) {
            throw new IllegalStateException("Sequence already has an executing Qwen chain");
        }
        if (this.terminalState != TerminalState.ACTIVE) {
            throw new IllegalStateException("Sequence is terminal: " + this.terminalState);
        }
        if (this.cancellationRequested) {
            throw new IllegalStateException("Sequence cancellation was requested");
        }
        if (expectedStartPosition != this.currentTokenPosition) {
            throw new IllegalArgumentException("Execution starts at " + expectedStartPosition + " but sequence is at "
                    + this.currentTokenPosition);
        }
        this.executionClaimed = true;
        this.activeLease = new ExecutionLease(this);
        return this.activeLease;
    }

    /// Releases the execution lease and publishes the next token position.
    public synchronized void releaseExecution(ExecutionLease lease, long nextTokenPosition) {
        requireLease(lease);
        if (nextTokenPosition < this.currentTokenPosition) {
            throw new IllegalArgumentException("nextTokenPosition cannot move backwards");
        }
        this.currentTokenPosition = nextTokenPosition;
        this.executionClaimed = false;
        this.activeLease.active = false;
        this.activeLease = null;
    }

    synchronized void markCancelled(ExecutionLease lease) {
        requireLease(lease);
        this.executionClaimed = false;
        this.cancellationRequested = true;
        this.terminalState = TerminalState.CANCELLED;
        this.activeLease.active = false;
        this.activeLease = null;
    }

    synchronized void markCancelledBeforeClaim() {
        if (this.executionClaimed) {
            throw new IllegalStateException("Sequence execution is already claimed");
        }
        this.cancellationRequested = true;
        this.terminalState = TerminalState.CANCELLED;
    }

    synchronized void markFailed(ExecutionLease lease, Throwable failure) {
        requireLease(lease);
        this.executionClaimed = false;
        this.terminalFailure = Objects.requireNonNull(failure, "failure");
        this.terminalState = TerminalState.FAILED;
        this.activeLease.active = false;
        this.activeLease = null;
    }

    synchronized void markFailedBeforeClaim(Throwable failure) {
        if (this.executionClaimed) {
            throw new IllegalStateException("Sequence execution is already claimed");
        }
        this.terminalFailure = Objects.requireNonNull(failure, "failure");
        this.terminalState = TerminalState.FAILED;
    }

    synchronized void markFailedAfterRelease(Throwable failure) {
        if (this.executionClaimed) {
            throw new IllegalStateException("Sequence execution is still claimed");
        }
        this.terminalFailure = Objects.requireNonNull(failure, "failure");
        this.terminalState = TerminalState.FAILED;
    }

    private void requireLease(ExecutionLease lease) {
        if (lease == null
                || lease.owner != this
                || !lease.active
                || !this.executionClaimed
                || this.activeLease != lease) {
            throw new IllegalStateException("Sequence mutation requires its active execution lease");
        }
    }
}
