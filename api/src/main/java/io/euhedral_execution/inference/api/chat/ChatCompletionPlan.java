package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.List;

/// A validated request, fully resolved before any session is opened.
/// `promptTokens + maxTokens` never exceeds the model context.
public record ChatCompletionPlan(
        String id,
        long created,
        String model,
        String prompt,
        int promptTokens,
        int maxTokens,
        GenerationConfig sampling,
        List<String> stops,
        boolean stream,
        boolean includeUsage) {

    public ChatCompletionPlan {
        stops = List.copyOf(stops);
    }
}
