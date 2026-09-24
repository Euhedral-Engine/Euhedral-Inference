package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/// Streaming `chat.completion.chunk`. `usage` appears only on the optional final usage chunk.
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ChatCompletionChunk(
        String id, String object, long created, String model, List<Choice> choices, Usage usage) {

    public static ChatCompletionChunk role(String id, long created, String model) {
        return choice(id, created, model, new Delta("assistant", ""), null);
    }

    public static ChatCompletionChunk content(String id, long created, String model, String text) {
        return choice(id, created, model, new Delta(null, text), null);
    }

    public static ChatCompletionChunk finish(String id, long created, String model, String finishReason) {
        return choice(id, created, model, new Delta(null, null), finishReason);
    }

    public static ChatCompletionChunk usage(String id, long created, String model, Usage usage) {
        return new ChatCompletionChunk(id, "chat.completion.chunk", created, model, List.of(), usage);
    }

    private static ChatCompletionChunk choice(String id, long created, String model, Delta delta, String finishReason) {
        return new ChatCompletionChunk(
                id, "chat.completion.chunk", created, model, List.of(new Choice(0, delta, null, finishReason)), null);
    }

    /// `logprobs` and `finish_reason` are serialized as explicit nulls, as OpenAI does.
    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Choice(
            int index,
            Delta delta,
            Object logprobs,
            @JsonProperty("finish_reason") String finishReason) {}

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record Delta(String role, String content) {}
}
