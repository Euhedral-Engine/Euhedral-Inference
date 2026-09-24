package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonProperty;

/// Tokenizer-derived token accounting for one completion.
public record Usage(
        @JsonProperty("prompt_tokens") int promptTokens,
        @JsonProperty("completion_tokens") int completionTokens,
        @JsonProperty("total_tokens") int totalTokens) {

    public static Usage of(int promptTokens, int completionTokens) {
        return new Usage(promptTokens, completionTokens, promptTokens + completionTokens);
    }
}
