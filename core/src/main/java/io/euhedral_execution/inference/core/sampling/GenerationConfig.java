package io.euhedral_execution.inference.core.sampling;

/// Immutable sampling-only settings for one generation.
/// A nonpositive temperature or greedy mode selects argmax; topK zero and topP one disable those filters.
public record GenerationConfig(float temperature, int topK, float topP, long seed, boolean greedy) {

    public GenerationConfig {
        if (!Float.isFinite(temperature)) throw new IllegalArgumentException("temperature must be finite");
        if (topK < 0) throw new IllegalArgumentException("topK must not be negative");
        if (!Float.isFinite(topP) || topP <= 0.0f || topP > 1.0f) {
            throw new IllegalArgumentException("topP must be finite and in the range (0, 1]");
        }
    }

    public static GenerationConfig greedy(long seed) {
        return new GenerationConfig(0.0f, 0, 1.0f, seed, true);
    }
}
