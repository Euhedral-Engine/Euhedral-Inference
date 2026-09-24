package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

/// The API layer's narrow view of the loaded model.
///
/// Production borrows the single `InferenceEngine`; tests substitute a scripted backend to exercise HTTP
/// behavior without CUDA. Implementations never expose engine internals to HTTP code.
public interface InferenceBackend {

    /// Public model ID accepted in requests and listed by `/v1/models`.
    String modelId();

    /// False once engine shutdown has begun; no further generation is admitted.
    boolean isAvailable();

    /// Maximum sequence length in tokens (prompt plus completion).
    int contextLength();

    /// Counts prompt tokens exactly as a new generation will encode the prompt.
    int countPromptTokens(String prompt);

    /// Opens one request-owned generation. The caller must close it on every path.
    ///
    /// @throws InferenceUnavailableException when the engine is closing or closed
    Generation openGeneration(GenerationConfig config);

    /// One request's sequence state. `cancel` may be called from any thread, including the output
    /// callback; `close` releases the sequence and waits for an in-flight quantum to detach.
    interface Generation extends AutoCloseable {

        /// Runs prefill and decode on the calling thread. `output` receives newly decoded, non-empty text
        /// on that same thread between quanta.
        Result generate(String prompt, int maxNewTokens, Consumer<String> output)
                throws InterruptedException, ExecutionException;

        void cancel();

        /// True when cancellation was requested by any party, including engine shutdown.
        boolean isCancelled();

        @Override
        void close();
    }

    /// `completionTokens` counts every sampled token, including a terminating stop token.
    record Result(int completionTokens, boolean stopTokenReached) {}
}
