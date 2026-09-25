package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/// Non-streaming `chat.completion` object with exactly one assistant choice. `content` is null when the
/// assistant only called tools; `tool_calls` is omitted when it called none.
public record ChatCompletionResponse(
        String id, String object, long created, String model, List<Choice> choices, Usage usage) {

    public static ChatCompletionResponse of(
            String id,
            long created,
            String model,
            String content,
            List<ToolCall> toolCalls,
            String finishReason,
            Usage usage) {
        AssistantMessage message = toolCalls.isEmpty()
                ? new AssistantMessage("assistant", content, null, null)
                : new AssistantMessage("assistant", content.isEmpty() ? null : content, null, List.copyOf(toolCalls));
        return new ChatCompletionResponse(
                id, "chat.completion", created, model, List.of(new Choice(0, message, null, finishReason)), usage);
    }

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record Choice(
            int index,
            AssistantMessage message,
            Object logprobs,
            @JsonProperty("finish_reason") String finishReason) {}

    @JsonInclude(JsonInclude.Include.ALWAYS)
    public record AssistantMessage(
            String role,
            String content,
            String refusal,

            @JsonInclude(JsonInclude.Include.NON_NULL) @JsonProperty("tool_calls")
            List<ToolCall> toolCalls) {}
}
