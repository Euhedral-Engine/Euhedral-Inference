package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.core.InferenceEngine;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

/// Borrows the Spring-owned `InferenceEngine`; the engine bean remains the terminal owner of model,
/// GPU, lattice, and runtime. Each generation wraps one engine-tracked `QwenGenerationSession`.
public final class EngineInferenceBackend implements InferenceBackend {
    private final InferenceEngine engine;
    private final String modelId;

    public EngineInferenceBackend(InferenceEngine engine, String modelId, QwenChatTemplate chatTemplate) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.modelId = Objects.requireNonNull(modelId, "modelId");
        // The formatter emits these as control tokens; plain-text encoding would silently corrupt prompts.
        for (String token : chatTemplate.controlTokens()) {
            if (engine.tokenizer().controlTokenId(token).isEmpty())
                throw new IllegalStateException("tokenizer lacks chat-template control token " + token);
        }
    }

    @Override
    public String modelId() {
        return this.modelId;
    }

    @Override
    public boolean isAvailable() {
        return !this.engine.isClosed();
    }

    @Override
    public int contextLength() {
        return this.engine.modelConfig().maxPositionEmbeddings();
    }

    @Override
    public int countPromptTokens(String prompt) {
        // A fresh session encodes its first prompt with the tokenizer's model special tokens.
        return this.engine.tokenizer().encodeWithModelSpecialTokens(prompt).length;
    }

    @Override
    public Generation openGeneration(GenerationConfig config) {
        if (this.engine.isClosed()) throw new InferenceUnavailableException("inference engine is shutting down");
        try {
            return new SessionGeneration(this.engine.createSession(config), this.engine.tokenizer());
        } catch (IllegalStateException closed) {
            // createSession's only state failure is closed admission; anything else is a real fault.
            if (this.engine.isClosed()) throw new InferenceUnavailableException("inference engine is shutting down");
            throw closed;
        }
    }

    private record SessionGeneration(QwenGenerationSession session, QwenTokenizer tokenizer) implements Generation {

        @Override
        public Result generate(String prompt, int maxNewTokens, Consumer<String> output)
                throws InterruptedException, ExecutionException {
            List<Integer> tokenIds = this.session.generate(prompt, maxNewTokens, output);
            boolean stopToken = !tokenIds.isEmpty() && this.tokenizer.isGenerationEosToken(tokenIds.getLast());
            return new Result(tokenIds.size(), stopToken);
        }

        @Override
        public void cancel() {
            this.session.cancel();
        }

        @Override
        public boolean isCancelled() {
            return this.session.isCancelled();
        }

        @Override
        public void close() {
            this.session.close();
        }
    }
}
