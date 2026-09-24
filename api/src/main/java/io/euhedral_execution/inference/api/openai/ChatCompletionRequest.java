package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

/// Supported subset of an OpenAI Chat Completions request.
///
/// `stop` is a string or a list of strings. Every other top-level field is captured in `otherFields`
/// and classified by `ChatRequestMapper` as ignorable metadata, a neutral default, or a rejected feature.
public record ChatCompletionRequest(
        String model,
        List<ChatMessage> messages,
        Boolean stream,
        @JsonProperty("stream_options") StreamOptions streamOptions,
        @JsonProperty("max_tokens") Integer maxTokens,
        @JsonProperty("max_completion_tokens") Integer maxCompletionTokens,
        Double temperature,
        @JsonProperty("top_p") Double topP,
        Long seed,
        Object stop,
        @JsonAnySetter Map<String, Object> otherFields) {

    public ChatCompletionRequest {
        otherFields = otherFields == null ? Map.of() : otherFields;
    }

    public record StreamOptions(
            @JsonProperty("include_usage") Boolean includeUsage,
            @JsonAnySetter Map<String, Object> otherFields) {

        public StreamOptions {
            otherFields = otherFields == null ? Map.of() : otherFields;
        }
    }
}
