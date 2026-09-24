package io.euhedral_execution.inference.api.chat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Renders system/user/assistant text turns in the ChatML-with-think format of the loaded Qwen checkpoint.
///
/// The checkpoint's Jinja template is the source of truth, but it relies on Python-specific Jinja
/// (reverse slicing, `str.startswith`, namespaces) that Java template engines do not reproduce. This class
/// therefore implements the template's text-only branch with `add_generation_prompt=true`,
/// `enable_thinking=false`, and no tools, and refuses at startup any template that no longer contains the
/// fragments it reproduces. Golden tests compare its output against the template rendered by jinja2.
public final class QwenChatTemplate {
    public static final String IM_START = "<|im_start|>";
    public static final String IM_END = "<|im_end|>";
    public static final String THINK_START = "<think>";
    public static final String THINK_END = "</think>";

    /// Exact Jinja source fragments whose behavior `render` reproduces. `\n` is the Jinja escape.
    private static final List<String> REQUIRED_FRAGMENTS = List.of(
            "{%- if messages[0].role == 'system' %}",
            "render_content(messages[0].content, false, true)|trim",
            "'<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')"
                    + "  + content + '<|im_end|>\\n'",
            "raise_exception('No user query found in messages.')",
            "render_content(message.content, true)|trim",
            "raise_exception('System message must be at the beginning.')",
            "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n'",
            "'<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content",
            "{%- if enable_thinking is undefined or enable_thinking is true %}",
            "{{- '<|im_start|>assistant\\n' }}",
            "{{- '<think>\\n\\n</think>\\n\\n' }}");

    private QwenChatTemplate() {}

    public enum Role {
        SYSTEM,
        USER,
        ASSISTANT
    }

    /// One text turn. `content` is the already-concatenated text of the message's text parts.
    public record Turn(Role role, String content) {
        public Turn {
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(content, "content");
        }
    }

    /// Thrown for conversations the checkpoint template itself rejects; messages mirror the template.
    public static final class InvalidConversationException extends IllegalArgumentException {
        InvalidConversationException(String message) {
            super(message);
        }
    }

    /// Loads `chat_template.jinja`, falling back to `tokenizer_config.json`'s `chat_template` string.
    public static QwenChatTemplate load(Path tokenizerDirectory) throws IOException {
        Path jinja = tokenizerDirectory.resolve("chat_template.jinja");
        if (Files.isRegularFile(jinja)) return fromTemplateSource(Files.readString(jinja));
        Path config = tokenizerDirectory.resolve("tokenizer_config.json");
        if (Files.isRegularFile(config)) {
            JsonNode template =
                    JsonMapper.shared().readTree(Files.readString(config)).path("chat_template");
            if (template.isString()) return fromTemplateSource(template.asString());
        }
        throw new IOException("checkpoint has no chat template in " + tokenizerDirectory);
    }

    /// Verifies that `source` is the Qwen template this formatter reproduces.
    public static QwenChatTemplate fromTemplateSource(String source) throws IOException {
        Objects.requireNonNull(source, "source");
        for (String fragment : REQUIRED_FRAGMENTS) {
            if (!source.contains(fragment))
                throw new IOException(
                        "unsupported checkpoint chat template; the Qwen formatter no longer matches fragment: "
                                + fragment);
        }
        return new QwenChatTemplate();
    }

    /// Control tokens this formatter emits; the tokenizer must encode each as one control token.
    public List<String> controlTokens() {
        return List.of(IM_START, IM_END, THINK_START, THINK_END);
    }

    /// Renders the conversation followed by the assistant generation prompt.
    public String render(List<Turn> turns) {
        Objects.requireNonNull(turns, "turns");
        if (turns.isEmpty()) throw new InvalidConversationException("No messages provided.");
        StringBuilder prompt = new StringBuilder();
        // Thinking is disabled, so reasoning instructions are empty and only a non-empty system prompt renders.
        if (turns.getFirst().role() == Role.SYSTEM) {
            String system = trim(turns.getFirst().content());
            if (!system.isEmpty())
                prompt.append(IM_START)
                        .append("system\n")
                        .append(system)
                        .append(IM_END)
                        .append('\n');
        }
        if (!hasUserQuery(turns)) throw new InvalidConversationException("No user query found in messages.");
        for (int index = 0; index < turns.size(); index++) {
            Turn turn = turns.get(index);
            String content = trim(turn.content());
            switch (turn.role()) {
                case SYSTEM -> {
                    if (index != 0) throw new InvalidConversationException("System message must be at the beginning.");
                }
                case USER ->
                    prompt.append(IM_START)
                            .append("user\n")
                            .append(content)
                            .append(IM_END)
                            .append('\n');
                // Without preserve_thinking the template keeps an empty think block on every assistant turn.
                case ASSISTANT ->
                    prompt.append(IM_START)
                            .append("assistant\n")
                            .append(THINK_START)
                            .append("\n\n")
                            .append(THINK_END)
                            .append("\n\n")
                            .append(content)
                            .append(IM_END)
                            .append('\n');
            }
        }
        prompt.append(IM_START)
                .append("assistant\n")
                .append(THINK_START)
                .append("\n\n")
                .append(THINK_END);
        return prompt.append("\n\n").toString();
    }

    /// Jinja's `trim` is Python `str.strip`, whose whitespace set includes NBSP and NEL unlike `String.strip`.
    static String trim(String text) {
        int start = 0;
        int end = text.length();
        while (start < end && isPythonSpace(text.charAt(start))) start++;
        while (end > start && isPythonSpace(text.charAt(end - 1))) end--;
        return text.substring(start, end);
    }

    private static boolean isPythonSpace(char value) {
        return Character.isWhitespace(value) || Character.isSpaceChar(value) || value == '\u0085';
    }

    /// Mirrors the template's search for a user turn that is not a wrapped tool response.
    private static boolean hasUserQuery(List<Turn> turns) {
        for (Turn turn : turns) {
            if (turn.role() != Role.USER) continue;
            String content = trim(turn.content());
            if (!(content.startsWith("<tool_response>") && content.endsWith("</tool_response>"))) return true;
        }
        return false;
    }
}
