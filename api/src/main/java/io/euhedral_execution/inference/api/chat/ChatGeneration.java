package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.api.openai.ToolCall;
import io.euhedral_execution.inference.api.openai.Usage;
import java.io.IOException;
import java.util.UUID;
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
final class ChatGeneration implements Runnable, ToolCallParser.Output {
    private static final Logger LOG = LoggerFactory.getLogger(ChatGeneration.class);

    private final ChatCompletionPlan plan;
    private final InferenceBackend.Generation generation;
    private final CompletionSink sink;
    private final StopSequenceFilter stopFilter;
    // Null when the request offers no callable tools: output is then plain text, byte for byte.
    private final JsonToolCallParser toolParser;
    // Set by container threads; read by the generation thread between quanta.
    private final AtomicBoolean abandoned = new AtomicBoolean();
    // Generation-thread confined.
    private boolean delivered;
    private ToolCallParser.MalformedToolCallException malformedToolCall;
    // Set when the single call allowed by parallel_tool_calls=false is complete.
    private boolean callLimitReached;

    ChatGeneration(ChatCompletionPlan plan, InferenceBackend.Generation generation, CompletionSink sink) {
        this.plan = plan;
        this.generation = generation;
        this.sink = sink;
        this.stopFilter = new StopSequenceFilter(plan.stops());
        this.toolParser = plan.tools().parsesOutput() ? new JsonToolCallParser(plan.tools()) : null;
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
            if (this.malformedToolCall != null) throw this.malformedToolCall;
            // Nobody in this request cancelled, so the engine did: it is shutting down. Sampled before the
            // flush below, which may itself match a stop sequence and cancel.
            boolean endedByRequest = this.stopFilter.matched() || this.callLimitReached;
            boolean engineCancelled = !endedByRequest && owned.isCancelled();
            if (!this.callLimitReached && !engineCancelled) {
                String remaining = this.stopFilter.finish();
                if (this.toolParser == null) emit(remaining);
                else {
                    this.toolParser.accept(remaining, this);
                    this.toolParser.finish(result.stopTokenReached() && !this.stopFilter.matched(), this);
                }
            }
            if (engineCancelled) {
                deliverFailure(OpenAiException.unavailable("Generation was interrupted by engine shutdown."));
                return;
            }
            String finishReason = finishReason(result);
            this.delivered = true;
            this.sink.finish(finishReason, Usage.of(this.plan.promptTokens(), result.completionTokens()));
        } catch (IOException clientGone) {
            abandon();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            deliverFailure(OpenAiException.unavailable("Generation was interrupted by server shutdown."));
        } catch (ToolCallParser.MalformedToolCallException malformed) {
            LOG.warn("Chat completion {} failed: {}", this.plan.id(), malformed.getMessage());
            deliverFailure(OpenAiException.invalidToolCall(malformed.getMessage()));
        } catch (ExecutionException | RuntimeException | Error failure) {
            if (!this.abandoned.get()) LOG.error("Chat completion {} failed", this.plan.id(), failure);
            deliverFailure(OpenAiException.serverError());
            if (failure instanceof Error error) throw error;
        }
    }

    private String finishReason(InferenceBackend.Result result) {
        if (this.stopFilter.matched()) return "stop";
        if (this.callLimitReached) return "tool_calls";
        if (!result.stopTokenReached()) return "length";
        return this.toolParser != null && this.toolParser.calls() > 0 ? "tool_calls" : "stop";
    }

    /// Output callback, invoked between quanta on this thread while the session holds its generation lock.
    private void onText(String text) {
        if (this.abandoned.get() || this.malformedToolCall != null || this.stopFilter.matched()) return;
        try {
            // Match stops on raw output before parsing a JSON tool-call envelope.
            String visible = this.stopFilter.accept(text);
            if (this.stopFilter.matched()) this.generation.cancel();
            if (this.toolParser == null) emit(visible);
            else this.toolParser.accept(visible, this);
        } catch (IOException clientGone) {
            abandon();
        } catch (ToolCallParser.MalformedToolCallException malformed) {
            // Fail closed: nothing after an invalid call can be returned, so stop decoding.
            this.malformedToolCall = malformed;
            this.generation.cancel();
        }
    }

    /// Assistant text after raw output has passed stop-sequence filtering.
    @Override
    public void content(String text) throws IOException {
        emit(text);
    }

    /// A complete call after raw output has passed stop-sequence filtering.
    @Override
    public void toolCall(String name, String arguments) throws IOException {
        String id = "call_" + UUID.randomUUID().toString().replace("-", "");
        this.sink.toolCall(this.toolParser.calls() - 1, ToolCall.function(id, name, arguments));
        if (!this.plan.tools().parallel()) {
            this.callLimitReached = true;
            this.generation.cancel();
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
