package io.euhedral_execution.inference.api.openai;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import java.util.Map;

/// One request message. `content` is a string, a list of content parts, or null.
/// `name` is accepted and ignored because the checkpoint template does not render participant names.
public record ChatMessage(
        String role,
        Object content,
        String name,
        @JsonAnySetter Map<String, Object> otherFields) {

    public ChatMessage {
        otherFields = otherFields == null ? Map.of() : otherFields;
    }
}
