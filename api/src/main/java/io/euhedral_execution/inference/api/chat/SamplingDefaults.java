package io.euhedral_execution.inference.api.chat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Checkpoint-recommended sampling from `generation_config.json`, used for fields a request omits.
/// `topK` has no OpenAI request field and always comes from here; zero disables it.
public record SamplingDefaults(float temperature, int topK, float topP) {

    public SamplingDefaults {
        if (!Float.isFinite(temperature) || temperature < 0.0f)
            throw new IllegalArgumentException("default temperature must be finite and non-negative");
        if (topK < 0) throw new IllegalArgumentException("default topK must not be negative");
        if (!Float.isFinite(topP) || topP <= 0.0f || topP > 1.0f)
            throw new IllegalArgumentException("default topP must be in the range (0, 1]");
    }

    /// Reads the checkpoint file; absent fields fall back to Hugging Face generation defaults.
    /// `do_sample: false` selects greedy decoding.
    public static SamplingDefaults load(Path tokenizerDirectory) throws IOException {
        Path path = tokenizerDirectory.resolve("generation_config.json");
        if (!Files.isRegularFile(path)) return new SamplingDefaults(1.0f, 0, 1.0f);
        JsonNode config = JsonMapper.shared().readTree(Files.readString(path));
        if (!config.path("do_sample").asBoolean(true)) return new SamplingDefaults(0.0f, 0, 1.0f);
        return new SamplingDefaults(
                (float) config.path("temperature").asDouble(1.0),
                config.path("top_k").asInt(0),
                (float) config.path("top_p").asDouble(1.0));
    }
}
