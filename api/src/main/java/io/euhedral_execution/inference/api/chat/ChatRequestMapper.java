package io.euhedral_execution.inference.api.chat;

import io.euhedral_execution.inference.api.engine.ApiProperties;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.ChatCompletionRequest;
import io.euhedral_execution.inference.api.openai.ChatMessage;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Predicate;
import org.springframework.stereotype.Component;

/// Validates an OpenAI request and resolves it into a `ChatCompletionPlan`.
///
/// Fields fall into four classes: supported; metadata that cannot change generation and is ignored;
/// behavioral features accepted only at their neutral value; and everything else, which is rejected.
@Component
public class ChatRequestMapper {
    static final int MAX_STOP_SEQUENCES = 4;

    /// Accepted and ignored: these never influence the generated tokens.
    private static final Set<String> IGNORED_METADATA =
            Set.of("user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key");

    /// Behavioral features that are not implemented; accepted only when the value requests no behavior.
    private static final Map<String, Predicate<Object>> NEUTRAL_VALUES = Map.ofEntries(
            Map.entry("n", value -> value instanceof Number number && number.doubleValue() == 1.0),
            Map.entry("logprobs", Boolean.FALSE::equals),
            Map.entry("top_logprobs", value -> false),
            Map.entry("frequency_penalty", ChatRequestMapper::isZero),
            Map.entry("presence_penalty", ChatRequestMapper::isZero),
            Map.entry("logit_bias", value -> value instanceof Map<?, ?> map && map.isEmpty()),
            Map.entry("tools", value -> value instanceof List<?> list && list.isEmpty()),
            Map.entry("functions", value -> value instanceof List<?> list && list.isEmpty()),
            Map.entry("tool_choice", "none"::equals),
            Map.entry("function_call", "none"::equals),
            // Inert without tools, which are rejected above.
            Map.entry("parallel_tool_calls", value -> value instanceof Boolean),
            Map.entry(
                    "response_format",
                    value -> value instanceof Map<?, ?> map && map.size() == 1 && "text".equals(map.get("type"))),
            Map.entry("modalities", List.of("text")::equals),
            Map.entry("reasoning_effort", value -> false),
            Map.entry("audio", value -> false),
            Map.entry("prediction", value -> false),
            Map.entry("web_search_options", value -> false),
            Map.entry("verbosity", value -> false));

    private static final Map<String, Predicate<Object>> MESSAGE_NEUTRAL_VALUES = Map.of(
            "tool_calls", value -> value instanceof List<?> list && list.isEmpty(),
            "refusal", value -> false,
            "tool_call_id", value -> false,
            "function_call", value -> false,
            "audio", value -> false,
            "reasoning_content", value -> false);

    private final InferenceBackend backend;
    private final QwenChatTemplate chatTemplate;
    private final SamplingDefaults samplingDefaults;
    private final ApiProperties apiProperties;

    public ChatRequestMapper(
            InferenceBackend backend,
            QwenChatTemplate chatTemplate,
            SamplingDefaults samplingDefaults,
            ApiProperties apiProperties) {
        this.backend = backend;
        this.chatTemplate = chatTemplate;
        this.samplingDefaults = samplingDefaults;
        this.apiProperties = apiProperties;
    }

    public ChatCompletionPlan plan(ChatCompletionRequest request) {
        if (request == null) throw OpenAiException.invalidRequest("Request body is required.", null);
        checkOtherFields(request.otherFields(), NEUTRAL_VALUES, "");
        boolean stream = Boolean.TRUE.equals(request.stream());
        boolean includeUsage = streamOptionsIncludeUsage(request, stream);
        requireServedModel(request.model());

        String prompt = renderPrompt(request.messages());
        int promptTokens = this.backend.countPromptTokens(prompt);
        int maxTokens = resolveMaxTokens(request, promptTokens);
        return new ChatCompletionPlan(
                "chatcmpl-" + UUID.randomUUID().toString().replace("-", ""),
                Instant.now().getEpochSecond(),
                this.backend.modelId(),
                prompt,
                promptTokens,
                maxTokens,
                sampling(request),
                stops(request.stop()),
                stream,
                includeUsage);
    }

    private void requireServedModel(String model) {
        if (model == null || model.isBlank())
            throw OpenAiException.invalidRequest("You must provide a model parameter.", "model");
        if (!model.equals(this.backend.modelId())) throw OpenAiException.modelNotFound(model);
    }

    private static boolean streamOptionsIncludeUsage(ChatCompletionRequest request, boolean stream) {
        var options = request.streamOptions();
        if (options == null) return false;
        if (!stream)
            throw OpenAiException.invalidRequest(
                    "The 'stream_options' parameter is only allowed when 'stream' is enabled.", "stream_options");
        for (var field : options.otherFields().entrySet()) {
            // Obfuscation padding only guards against network side channels; omitting it changes no content.
            if (field.getKey().equals("include_obfuscation")) continue;
            throw OpenAiException.unrecognizedArgument("stream_options." + field.getKey());
        }
        return Boolean.TRUE.equals(options.includeUsage());
    }

    private String renderPrompt(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty())
            throw OpenAiException.invalidRequest("'messages' must contain at least one message.", "messages");
        List<QwenChatTemplate.Turn> turns = new ArrayList<>(messages.size());
        for (int index = 0; index < messages.size(); index++) {
            ChatMessage message = messages.get(index);
            String path = "messages[" + index + "]";
            if (message == null) throw OpenAiException.invalidRequest("Message must be an object.", path);
            checkOtherFields(message.otherFields(), MESSAGE_NEUTRAL_VALUES, path + ".");
            QwenChatTemplate.Role role = role(message.role(), path);
            turns.add(new QwenChatTemplate.Turn(role, content(message.content(), role, path)));
        }
        try {
            return this.chatTemplate.render(turns);
        } catch (QwenChatTemplate.InvalidConversationException invalid) {
            throw OpenAiException.invalidRequest(invalid.getMessage(), "messages");
        }
    }

    private static QwenChatTemplate.Role role(String role, String path) {
        if (role == null) throw OpenAiException.invalidRequest("Message role is required.", path + ".role");
        return switch (role) {
            // OpenAI's newer name for instructions formerly sent as system messages.
            case "system", "developer" -> QwenChatTemplate.Role.SYSTEM;
            case "user" -> QwenChatTemplate.Role.USER;
            case "assistant" -> QwenChatTemplate.Role.ASSISTANT;
            case "tool", "function" ->
                throw OpenAiException.invalidRequest(
                        "Message role '" + role + "' is not supported; tool calling is not implemented.",
                        path + ".role");
            default -> throw OpenAiException.invalidRequest("Invalid message role '" + role + "'.", path + ".role");
        };
    }

    /// Mirrors the template's `render_content` for text: strings pass through, text parts concatenate.
    private static String content(Object content, QwenChatTemplate.Role role, String path) {
        if (content == null) {
            if (role == QwenChatTemplate.Role.ASSISTANT) return "";
            throw OpenAiException.invalidRequest("Message content is required.", path + ".content");
        }
        if (content instanceof String text) return text;
        if (!(content instanceof List<?> parts))
            throw OpenAiException.invalidRequest(
                    "Message content must be a string or an array of content parts.", path + ".content");
        StringBuilder text = new StringBuilder();
        for (int index = 0; index < parts.size(); index++) {
            String partPath = path + ".content[" + index + "]";
            if (!(parts.get(index) instanceof Map<?, ?> part))
                throw OpenAiException.invalidRequest("Content part must be an object.", partPath);
            Object type = part.get("type");
            if (!"text".equals(type)) {
                if (type instanceof String) throw OpenAiException.unsupportedParameter(partPath + ".type=" + type);
                throw OpenAiException.invalidRequest("Content part type is required.", partPath + ".type");
            }
            if (!(part.get("text") instanceof String partText))
                throw OpenAiException.invalidRequest("Text content part requires 'text'.", partPath + ".text");
            for (Object key : part.keySet()) {
                if (!key.equals("type") && !key.equals("text"))
                    throw OpenAiException.unrecognizedArgument(partPath + "." + key);
            }
            text.append(partText);
        }
        return text.toString();
    }

    private int resolveMaxTokens(ChatCompletionRequest request, int promptTokens) {
        Integer requested = request.maxCompletionTokens();
        String param = "max_completion_tokens";
        if (request.maxTokens() != null) {
            if (requested != null && !requested.equals(request.maxTokens()))
                throw OpenAiException.invalidRequest(
                        "'max_tokens' and 'max_completion_tokens' conflict; send only one.", "max_tokens");
            requested = request.maxTokens();
            param = "max_tokens";
        }
        int remaining = this.backend.contextLength() - promptTokens;
        if (requested != null) {
            if (requested < 1) throw OpenAiException.invalidRequest("'" + param + "' must be at least 1.", param);
            if (requested > remaining)
                throw OpenAiException.contextLengthExceeded(
                        "This model's maximum context length is " + this.backend.contextLength()
                                + " tokens. However, you requested " + (promptTokens + requested)
                                + " tokens (" + promptTokens + " in the messages, " + requested
                                + " in the completion).",
                        param);
            return requested;
        }
        if (remaining < 1)
            throw OpenAiException.contextLengthExceeded(
                    "This model's maximum context length is " + this.backend.contextLength()
                            + " tokens, but the messages use " + promptTokens + " tokens.",
                    "messages");
        return Math.min(this.apiProperties.defaultMaxTokens(), remaining);
    }

    /// Omitted fields take the checkpoint's generation_config values; temperature or top_p of 0 is greedy.
    private GenerationConfig sampling(ChatCompletionRequest request) {
        float temperature = this.samplingDefaults.temperature();
        if (request.temperature() != null) {
            double value = request.temperature();
            if (!(value >= 0.0 && value <= 2.0))
                throw OpenAiException.invalidRequest("'temperature' must be between 0 and 2.", "temperature");
            temperature = (float) value;
        }
        float topP = this.samplingDefaults.topP();
        if (request.topP() != null) {
            double value = request.topP();
            if (!(value >= 0.0 && value <= 1.0))
                throw OpenAiException.invalidRequest("'top_p' must be between 0 and 1.", "top_p");
            topP = (float) value;
        }
        long seed = request.seed() != null
                ? request.seed()
                : ThreadLocalRandom.current().nextLong();
        boolean greedy = temperature == 0.0f || topP == 0.0f;
        return new GenerationConfig(temperature, this.samplingDefaults.topK(), greedy ? 1.0f : topP, seed, greedy);
    }

    private static List<String> stops(Object stop) {
        if (stop == null) return List.of();
        List<?> values = stop instanceof String single ? List.of(single) : stop instanceof List<?> list ? list : null;
        if (values == null)
            throw OpenAiException.invalidRequest("'stop' must be a string or an array of strings.", "stop");
        if (values.size() > MAX_STOP_SEQUENCES)
            throw OpenAiException.invalidRequest(
                    "'stop' accepts at most " + MAX_STOP_SEQUENCES + " sequences.", "stop");
        List<String> stops = new ArrayList<>(values.size());
        for (Object value : values) {
            if (!(value instanceof String text) || text.isEmpty())
                throw OpenAiException.invalidRequest("Each stop sequence must be a non-empty string.", "stop");
            stops.add(text);
        }
        return stops;
    }

    private static void checkOtherFields(
            Map<String, Object> fields, Map<String, Predicate<Object>> neutralValues, String prefix) {
        for (var field : fields.entrySet()) {
            String name = field.getKey();
            if (prefix.isEmpty() && IGNORED_METADATA.contains(name)) continue;
            Predicate<Object> neutral = neutralValues.get(name);
            if (neutral == null) throw OpenAiException.unrecognizedArgument(prefix + name);
            if (field.getValue() != null && !neutral.test(field.getValue()))
                throw OpenAiException.unsupportedParameter(prefix + name);
        }
    }

    private static boolean isZero(Object value) {
        return value instanceof Number number && number.doubleValue() == 0.0;
    }
}
