package io.euhedral_execution.inference.api.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Compares the Java formatter with the checkpoint template as rendered by jinja2.
///
/// `qwen-chat-template.jinja` is the checkpoint's `chat_template.jinja` (Apache-2.0).
/// `qwen-chat-template-golden.json` was produced by rendering it with jinja2 3.1 in a sandboxed environment
/// using `trim_blocks`/`lstrip_blocks` (as Hugging Face does), `add_generation_prompt=True`, and
/// `enable_thinking=False`.
class QwenChatTemplateTest {

    @Test
    void rendersExactlyWhatTheCheckpointTemplateRenders() throws IOException {
        QwenChatTemplate template = QwenChatTemplate.fromTemplateSource(resource("/qwen-chat-template.jinja"));
        JsonNode golden = JsonMapper.shared().readTree(resource("/qwen-chat-template-golden.json"));
        int compared = 0;
        for (JsonNode example : golden.get("cases")) {
            String name = example.get("name").asString();
            assertEquals(example.get("expected").asString(), template.render(turns(example)), name);
            compared++;
        }
        assertEquals(10, compared);
    }

    @Test
    void rejectsConversationsTheCheckpointTemplateRejects() throws IOException {
        QwenChatTemplate template = QwenChatTemplate.fromTemplateSource(resource("/qwen-chat-template.jinja"));
        JsonNode golden = JsonMapper.shared().readTree(resource("/qwen-chat-template-golden.json"));
        for (JsonNode example : golden.get("errors")) {
            var failure = assertThrows(
                    QwenChatTemplate.InvalidConversationException.class, () -> template.render(turns(example)));
            assertEquals(example.get("message").asString(), failure.getMessage());
        }
        assertThrows(QwenChatTemplate.InvalidConversationException.class, () -> template.render(List.of()));
    }

    @Test
    void refusesATemplateWhoseFormatItDoesNotReproduce() throws IOException {
        String source = resource("/qwen-chat-template.jinja");
        String changed = source.replace("'<think>\\n\\n</think>\\n\\n'", "'<think>\\n</think>\\n'");
        var failure = assertThrows(IOException.class, () -> QwenChatTemplate.fromTemplateSource(changed));
        assertTrue(failure.getMessage().contains("unsupported checkpoint chat template"));
    }

    @Test
    void loadsFromJinjaFileOrTokenizerConfig(@TempDir Path directory) throws IOException {
        String source = resource("/qwen-chat-template.jinja");
        assertThrows(IOException.class, () -> QwenChatTemplate.load(directory));
        Files.writeString(
                directory.resolve("tokenizer_config.json"),
                JsonMapper.shared().writeValueAsString(java.util.Map.of("chat_template", source)));
        QwenChatTemplate.load(directory);
        Files.writeString(directory.resolve("chat_template.jinja"), "{{ messages }}");
        assertThrows(IOException.class, () -> QwenChatTemplate.load(directory), "the .jinja file takes precedence");
    }

    /// Converts golden OpenAI-style messages the same way `ChatRequestMapper` does for text content.
    private static List<QwenChatTemplate.Turn> turns(JsonNode example) {
        List<QwenChatTemplate.Turn> turns = new ArrayList<>();
        for (JsonNode message : example.get("messages")) {
            JsonNode content = message.get("content");
            StringBuilder text = new StringBuilder();
            if (content.isString()) text.append(content.asString());
            else if (content.isArray())
                for (JsonNode part : content) text.append(part.get("text").asString());
            turns.add(new QwenChatTemplate.Turn(
                    QwenChatTemplate.Role.valueOf(message.get("role").asString().toUpperCase(java.util.Locale.ROOT)),
                    text.toString()));
        }
        return turns;
    }

    private static String resource(String name) throws IOException {
        try (var input = QwenChatTemplateTest.class.getResourceAsStream(name)) {
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
