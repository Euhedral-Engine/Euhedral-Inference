package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.api.openai.Usage;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// Runs one request's generation on a generation-executor thread and closes its session.
///
/// Ownership: the request thread opens the generation and transfers it here when the executor accepts
/// this task; on rejection or executor shutdown `discard` closes it instead. `run` is the terminal owner
/// otherwise and closes the session on every path. `abandon` is the only cross-thread entry point: the
/// container calls it on disconnect or timeout, and it cancels the session so no further quantum starts.
final class ChatGeneration implements Runnable {
    private static final Logger LOG = LoggerFactory.getLogger(ChatGeneration.class);

    private final ChatCompletionPlan plan;
    private final InferenceBackend.Generation generation;
    private final CompletionSink sink;
    private final StopSequenceFilter stopFilter;
    // Set by container threads; read by the generation thread between quanta.
    private final AtomicBoolean abandoned = new AtomicBoolean();
    // Generation-thread confined.
    private boolean delivered;

    ChatGeneration(ChatCompletionPlan plan, InferenceBackend.Generation generation, CompletionSink sink) {
        this.plan = plan;
        this.generation = generation;
        this.sink = sink;
        this.stopFilter = new StopSequenceFilter(plan.stops());
    }

    /// The client disconnected or timed out: stop generating. Safe from any thread and after completion.
    void abandon() {
        if (this.abandoned.compareAndSet(false, true)) this.generation.cancel();
    }

    boolean isAbandoned() {
        return this.abandoned.get();
    }

    /// Releases a task that will never run (rejected or dropped from the queue at shutdown).
    void discard(OpenAiException reason) {
        this.abandoned.set(true);
        try {
            this.generation.close();
        } finally {
            this.sink.fail(reason);
        }
    }

    @Override
    public void run() {
        try (InferenceBackend.Generation owned = this.generation) {
            if (this.abandoned.get()) return;
            this.sink.start();
            InferenceBackend.Result result = owned.generate(this.plan.prompt(), this.plan.maxTokens(), this::onText);
            if (this.abandoned.get()) return;
            if (!this.stopFilter.matched()) {
                emit(this.stopFilter.finish());
                // Nobody in this request cancelled, so the engine did: it is shutting down.
                if (owned.isCancelled()) {
                    deliverFailure(OpenAiException.unavailable("Generation was interrupted by engine shutdown."));
                    return;
                }
            }
            String finishReason = this.stopFilter.matched() || result.stopTokenReached() ? "stop" : "length";
            this.delivered = true;
            this.sink.finish(finishReason, Usage.of(this.plan.promptTokens(), result.completionTokens()));
        } catch (IOException clientGone) {
            abandon();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            deliverFailure(OpenAiException.unavailable("Generation was interrupted by server shutdown."));
        } catch (ExecutionException | RuntimeException | Error failure) {
            if (!this.abandoned.get()) LOG.error("Chat completion {} failed", this.plan.id(), failure);
            deliverFailure(OpenAiException.serverError());
            if (failure instanceof Error error) throw error;
        }
    }

    /// Output callback, invoked between quanta on this thread while the session holds its generation lock.
    private void onText(String text) {
        if (this.abandoned.get()) return;
        String visible = this.stopFilter.accept(text);
        // Cancelling here prevents the next decode quantum from being admitted.
        if (this.stopFilter.matched()) this.generation.cancel();
        try {
            emit(visible);
        } catch (IOException clientGone) {
            abandon();
        }
    }

    private void emit(String text) throws IOException {
        if (!text.isEmpty()) this.sink.text(text);
    }

    private void deliverFailure(OpenAiException error) {
        if (this.abandoned.get() || this.delivered) return;
        this.delivered = true;
        this.sink.fail(error);
    }
}
