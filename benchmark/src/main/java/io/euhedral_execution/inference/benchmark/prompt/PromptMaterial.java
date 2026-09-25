package io.euhedral_execution.inference.benchmark.prompt;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToIntFunction;

/// Deterministic raw-text prompt sized in actual tokenizer tokens.
///
/// Text is a space-separated sequence of words drawn from [#WORDS] by `SplittableRandom(seed)`.
/// The longest word prefix whose token count does not exceed the target is kept; single
/// punctuation suffixes are then tried to reach the target exactly. `actualTokens` is the count
/// the session itself will encode (model special tokens included); it may be below the target when
/// no suffix fits. The same generator, seed, target, and tokenizer reproduce the same `sha256`.
/// Prompts are not chat-templated.
public record PromptMaterial(String text, int targetTokens, int actualTokens, String sha256, String generator) {
    public static final String GENERATOR = "euhedral-words-v1";
    public static final long DEFAULT_SEED = 20260925L;

    static final List<String> WORDS = List.of(
            "the",
            "river",
            "carries",
            "stone",
            "across",
            "a",
            "quiet",
            "valley",
            "where",
            "engineers",
            "measure",
            "light",
            "and",
            "time",
            "while",
            "clocks",
            "record",
            "every",
            "small",
            "change",
            "in",
            "pressure",
            "under",
            "bright",
            "northern",
            "sky",
            "people",
            "build",
            "bridges",
            "from",
            "steel",
            "glass",
            "wood",
            "maps",
            "show",
            "old",
            "roads",
            "that",
            "follow",
            "water",
            "toward",
            "distant",
            "cities",
            "markets",
            "sell",
            "bread",
            "salt",
            "copper",
            "tools",
            "children",
            "learn",
            "numbers",
            "letters",
            "songs",
            "winter",
            "brings",
            "snow",
            "summer",
            "harvest",
            "machines",
            "turn",
            "slowly",
            "through",
            "night");
    private static final List<String> SUFFIXES = List.of(".", ",", ";", "!", "?", ":", " .", " ,");

    public static PromptMaterial build(int targetTokens, long seed, ToIntFunction<String> tokenCount) {
        if (targetTokens <= 0) throw new IllegalArgumentException("targetTokens must be positive");
        String[] words = new String[targetTokens * 2 + 8];
        SplittableRandom random = new SplittableRandom(seed);
        for (int index = 0; index < words.length; index++) words[index] = WORDS.get(random.nextInt(WORDS.size()));

        int low = 1;
        int high = words.length;
        if (tokenCount.applyAsInt(join(words, low)) > targetTokens)
            throw new IllegalArgumentException("target " + targetTokens + " is below the shortest prompt");
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            if (tokenCount.applyAsInt(join(words, middle)) <= targetTokens) low = middle;
            else high = middle - 1;
        }
        String text = join(words, low);
        int count = tokenCount.applyAsInt(text);
        while (count < targetTokens) {
            String extended = null;
            int extendedCount = count;
            for (String suffix : SUFFIXES) {
                int candidate = tokenCount.applyAsInt(text + suffix);
                if (candidate > count && candidate <= targetTokens) {
                    extended = text + suffix;
                    extendedCount = candidate;
                    break;
                }
            }
            if (extended == null) break;
            text = extended;
            count = extendedCount;
        }
        return new PromptMaterial(text, targetTokens, count, sha256(text), GENERATOR);
    }

    private static String join(String[] words, int count) {
        return String.join(" ", java.util.Arrays.asList(words).subList(0, count));
    }

    public static String sha256(String text) {
        try {
            return HexFormat.of()
                    .formatHex(MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }
}
