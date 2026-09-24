package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.IncrementalDecoder;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

/// Coordinates prompt and decode quanta for one persistent Qwen sequence.
///
/// The tokenizer, plan, runtime, and GPU are borrowed. The session owns its sequence and one sampler;
/// each completed prompt-to-output stream is flushed before the decoder is replaced for a later prompt.
public final class QwenGenerationSession implements AutoCloseable {

    private final QwenTokenizer tokenizer;
    private final QwenExecutionPlan plan;
    private final EuhedralInferenceRuntime runtime;
    private final ExecutionGpu gpu;
    private final QwenSequenceState sequence;
    private final QwenLogitsSampler sampler;
    private final List<Integer> generatedTokenIds = new ArrayList<>();
    private final AtomicBoolean generationActive = new AtomicBoolean();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final ReentrantLock generationLock = new ReentrantLock();
    private Consumer<? super QwenGenerationSession> closeListener;

    private IncrementalDecoder decoder;
    private boolean decoderFinished;
    private boolean promptPrefilled;

    /// Creates a session with a new persistent sequence owned by this instance.
    /// The plan, runtime, GPU, and tokenizer are borrowed and must remain usable until the session is closed.
    public QwenGenerationSession(
            QwenTokenizer tokenizer,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config) {
        this(tokenizer, plan, runtime, gpu, sequenceId, config, ignored -> {});
    }

    /// Creates an owner-tracked session. The listener runs after successful cleanup with generation
    /// stopped, under the session lifecycle lock; it must not block or invoke session operations.
    /// Failed cleanup retains the listener for retry. Successful notification releases its reference.
    public QwenGenerationSession(
            QwenTokenizer tokenizer,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime,
            ExecutionGpu gpu,
            long sequenceId,
            GenerationConfig config,
            Consumer<? super QwenGenerationSession> closeListener) {
        this.closeListener = Objects.requireNonNull(closeListener, "closeListener");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        Objects.requireNonNull(config, "config");
        this.sequence = new QwenSequenceState(sequenceId);
        this.sampler = new QwenLogitsSampler(config, plan.weights().config().vocabSize());
        this.decoder = tokenizer.newIncrementalDecoder();
    }

    /// Prefills a prompt at the current sequence position and returns the IDs sampled by this call.
    /// The first prompt uses configured model special tokens; continuation prompts encode only their text.
    /// The output callback receives only newly decoded text and is never called for empty chunks.
    /// A sampled generation terminator is included in the returned IDs but is not sent through decode.
    public List<Integer> generate(String prompt, int maxNewTokens, Consumer<String> output)
            throws InterruptedException, ExecutionException {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(output, "output");
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must not be negative");
        if (!this.generationActive.compareAndSet(false, true)) {
            throw new IllegalStateException("a generation is already active for this Qwen session");
        }
        if (!this.generationLock.tryLock()) {
            this.generationActive.set(false);
            throw new IllegalStateException("the Qwen session is changing lifecycle state");
        }

        try {
            ensureUsable();
            if (this.decoderFinished) {
                this.decoder = this.tokenizer.newIncrementalDecoder();
                this.decoderFinished = false;
            }
            int[] promptTokenIds = this.promptPrefilled
                    ? this.tokenizer.encodeText(prompt)
                    : this.tokenizer.encodeWithModelSpecialTokens(prompt);
            if (promptTokenIds.length == 0) {
                throw new IllegalArgumentException("prompt must encode to at least one token");
            }

            try {
                return generateLocked(promptTokenIds, maxNewTokens, output);
            } catch (InterruptedException | ExecutionException | RuntimeException | Error failure) {
                if (this.sequence.terminalState() == QwenSequenceState.TerminalState.ACTIVE) {
                    try {
                        requestCancellation();
                    } catch (RuntimeException | Error cleanupFailure) {
                        if (cleanupFailure != failure) failure.addSuppressed(cleanupFailure);
                    }
                }
                throw failure;
            }
        } finally {
            this.generationActive.set(false);
            try {
                if (this.closed.get()) completeClose();
            } finally {
                this.generationLock.unlock();
            }
        }
    }

    /// Requests cancellation of the current quantum or prevents the next one from starting.
    /// A cancelled session cannot accept another prompt and remains owned until closed.
    public void cancel() {
        if (this.closed.get()) return;
        requestCancellation();
    }

    /// Returns the authoritative token position retained by the session's sequence state.
    public long currentTokenPosition() {
        return this.sequence.currentTokenPosition();
    }

    /// Returns an immutable snapshot of tokens sampled by all prompts in this session.
    public List<Integer> generatedTokenIds() {
        synchronized (this.generatedTokenIds) {
            return List.copyOf(this.generatedTokenIds);
        }
    }

    /// Reports whether cancellation has been requested for this session.
    public boolean isCancelled() {
        return this.cancelled.get();
    }

    /// Reports whether the caller has closed this session.
    public boolean isClosed() {
        return this.closed.get();
    }

    /// Allows the owning engine to reject reentrant shutdown from an output callback.
    public boolean isGeneratingOnCurrentThread() {
        return this.generationLock.isHeldByCurrentThread();
    }

    QwenSequenceState sequenceState() {
        return this.sequence;
    }

    /// Completes the sequence, releasing its persistent KV and GDN state.
    /// Closing during generation requests cancellation and waits for the in-flight quantum to detach.
    @Override
    public void close() {
        boolean firstClose = this.closed.compareAndSet(false, true);
        if (firstClose && this.generationActive.get()) requestCancellation();
        if (this.generationActive.get() && this.generationLock.isHeldByCurrentThread()) return;
        this.generationLock.lock();
        try {
            completeClose();
        } finally {
            this.generationLock.unlock();
        }
    }

    private void completeClose() {
        this.sequence.complete();
        if (this.closeListener != null) {
            this.closeListener.accept(this);
            this.closeListener = null;
        }
    }

    private List<Integer> generateLocked(int[] promptTokenIds, int maxNewTokens, Consumer<String> output)
            throws InterruptedException, ExecutionException {
        List<Integer> callTokenIds = new ArrayList<>();
        if (isStopRequested()) return List.of();

        QwenExecutionContext prefill = new QwenExecutionContext(
                this.plan,
                this.sequence,
                QwenExecutionContext.ExecutionKind.PREFILL,
                this.sequence.currentTokenPosition(),
                promptTokenIds);
        OptionalInt nextToken = executeAndSelect(prefill, maxNewTokens > 0);
        if (isStopRequested()) return List.of();
        this.promptPrefilled = true;
        if (maxNewTokens == 0) {
            finishDecoder(output);
            return List.of();
        }

        boolean endedNormally = false;
        for (int generated = 0; generated < maxNewTokens; generated++) {
            if (isStopRequested() || nextToken.isEmpty()) break;
            int tokenId = nextToken.getAsInt();
            if (isStopRequested()) break;
            if (this.tokenizer.isGenerationEosToken(tokenId)) {
                callTokenIds.add(tokenId);
                synchronized (this.generatedTokenIds) {
                    this.generatedTokenIds.add(tokenId);
                }
                // EOS terminates the generation and is not a model input quantum.
                endedNormally = true;
                break;
            }
            if (isStopRequested()) break;
            boolean anotherTokenAllowed = generated + 1 < maxNewTokens;
            callTokenIds.add(tokenId);
            synchronized (this.generatedTokenIds) {
                this.generatedTokenIds.add(tokenId);
            }
            emit(output, this.decoder.append(tokenId));
            // Cancellation makes the session terminal; do not admit another quantum to preserve history.
            if (isStopRequested()) break;
            QwenExecutionContext decode = new QwenExecutionContext(
                    this.plan,
                    this.sequence,
                    QwenExecutionContext.ExecutionKind.DECODE,
                    this.sequence.currentTokenPosition(),
                    new int[] {tokenId});
            // Commit the final non-terminal token for continuation without sampling beyond the limit.
            nextToken = executeAndSelect(decode, anotherTokenAllowed);
            if (!anotherTokenAllowed) endedNormally = !isStopRequested();
            if (isStopRequested()) break;
            if (!anotherTokenAllowed) break;
        }
        if (endedNormally && !isStopRequested()) finishDecoder(output);
        return List.copyOf(callTokenIds);
    }

    private OptionalInt executeAndSelect(QwenExecutionContext context, boolean selectToken)
            throws InterruptedException, ExecutionException {
        QwenDeviceLogits logits = null;
        Throwable executionFailure = null;
        try {
            List<QwenExecutionContext.Outcome> outcomes = this.runtime.execute(List.of(context));
            if (outcomes.size() != 1) {
                throw new IllegalStateException("runtime returned an unexpected number of quantum outcomes");
            }
            QwenExecutionContext.Outcome outcome = outcomes.getFirst();
            if (outcome.status() == QwenExecutionContext.Status.CANCELLED) {
                this.cancelled.set(true);
                return OptionalInt.empty();
            }
            if (outcome.status() == QwenExecutionContext.Status.FAILED) {
                throw new IllegalStateException("Qwen execution quantum failed", outcome.failure());
            }
            if (outcome.status() != QwenExecutionContext.Status.SUCCESS) {
                throw new IllegalStateException("runtime returned an unknown quantum outcome");
            }
            if (!selectToken || isStopRequested()) return OptionalInt.empty();
            logits = context.logitsOutput()
                    .orElseThrow(() -> new IllegalStateException("successful Qwen quantum produced no logits"));
            return OptionalInt.of(this.sampler.selectToken(logits, this.gpu));
        } catch (InterruptedException | ExecutionException | RuntimeException | Error failure) {
            executionFailure = failure;
            throw failure;
        } finally {
            if (logits == null) logits = context.logitsOutput().orElse(null);
            if (logits != null) {
                try {
                    logits.close();
                } catch (RuntimeException | Error cleanupFailure) {
                    if (executionFailure != null) executionFailure.addSuppressed(cleanupFailure);
                    else throw cleanupFailure;
                }
            }
        }
    }

    private void finishDecoder(Consumer<String> output) {
        if (this.decoderFinished) return;
        String remaining = this.decoder.finish();
        this.decoderFinished = true;
        emit(output, remaining);
    }

    private static void emit(Consumer<String> output, String text) {
        if (!text.isEmpty()) output.accept(text);
    }

    private void ensureUsable() {
        if (this.closed.get()) throw new IllegalStateException("Qwen generation session is closed");
        if (this.cancelled.get()) throw new IllegalStateException("Qwen generation session is cancelled");
        if (this.sequence.terminalState() != QwenSequenceState.TerminalState.ACTIVE) {
            throw new IllegalStateException("Qwen sequence is terminal: " + this.sequence.terminalState());
        }
    }

    private boolean isStopRequested() {
        return this.cancelled.get() || this.closed.get() || this.sequence.cancellationRequested();
    }

    private void requestCancellation() {
        if (this.cancelled.compareAndSet(false, true)) this.sequence.cancel();
    }
}
