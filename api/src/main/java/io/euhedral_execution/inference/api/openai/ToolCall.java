package io.euhedral_execution.inference.api.openai;

/// A function call the assistant made. `arguments` is a JSON object serialized as a string, as in OpenAI.
public record ToolCall(String id, String type, Function function) {

    public static ToolCall function(String id, String name, String arguments) {
        return new ToolCall(id, "function", new Function(name, arguments));
    }

    public record Function(String name, String arguments) {}
}
