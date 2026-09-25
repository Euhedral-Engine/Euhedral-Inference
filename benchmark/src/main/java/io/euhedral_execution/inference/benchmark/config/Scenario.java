package io.euhedral_execution.inference.benchmark.config;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/// One benchmark workload: a raw-text prompt of `targetPromptTokens` tokenizer tokens followed by
/// at most `requestedNewTokens` generated tokens in a fresh session.
///
/// Syntax (also its JSON form): `prefill:P`, `first-token:P`, `prompt-to-n:P:N`, `decode:P:N`.
public record Scenario(Kind kind, int targetPromptTokens, int requestedNewTokens) {
    public enum Kind {
        /// Prompt prefill only (`maxNewTokens = 0`); no token is sampled, so there is no first-token time.
        PREFILL("prefill"),
        /// Prefill plus the first sampled token (`maxNewTokens = 1`).
        FIRST_TOKEN("first-token"),
        /// Prompt to the Nth selected token.
        PROMPT_TO_N("prompt-to-n"),
        /// Sustained greedy-or-sampled decode after a short prompt; eligible only if all N tokens are produced.
        SUSTAINED_DECODE("decode");

        final String label;

        Kind(String label) {
            this.label = label;
        }

        /// The kind's name in scenario specifications and result rows.
        public String label() {
            return this.label;
        }
    }

    /// Covers short through large prefill, first token, prompt-to-N, and steady-state decode.
    public static final List<Scenario> DEFAULT_SUITE = List.of(
            new Scenario(Kind.PREFILL, 32, 0),
            new Scenario(Kind.PREFILL, 256, 0),
            new Scenario(Kind.PREFILL, 1024, 0),
            new Scenario(Kind.PREFILL, 4096, 0),
            new Scenario(Kind.FIRST_TOKEN, 1024, 1),
            new Scenario(Kind.PROMPT_TO_N, 1024, 64),
            new Scenario(Kind.SUSTAINED_DECODE, 32, 256));

    public Scenario {
        Objects.requireNonNull(kind, "kind");
        if (targetPromptTokens <= 0) throw new IllegalArgumentException("scenario prompt tokens must be positive");
        switch (kind) {
            case PREFILL -> {
                if (requestedNewTokens != 0) throw new IllegalArgumentException("prefill scenarios generate no tokens");
            }
            case FIRST_TOKEN -> {
                if (requestedNewTokens != 1) throw new IllegalArgumentException("first-token scenarios generate one");
            }
            case PROMPT_TO_N, SUSTAINED_DECODE -> {
                if (requestedNewTokens < 2)
                    throw new IllegalArgumentException(kind.label + " scenarios need at least two new tokens");
            }
        }
    }

    public String name() {
        return switch (this.kind) {
            case PREFILL, FIRST_TOKEN -> this.kind.label + "-" + this.targetPromptTokens;
            case PROMPT_TO_N, SUSTAINED_DECODE ->
                this.kind.label + "-" + this.targetPromptTokens + "-" + this.requestedNewTokens;
        };
    }

    /// Returns this scenario's parseable specification, which is also its JSON value.
    @JsonValue
    public String spec() {
        return switch (this.kind) {
            case PREFILL, FIRST_TOKEN -> this.kind.label + ":" + this.targetPromptTokens;
            case PROMPT_TO_N, SUSTAINED_DECODE ->
                this.kind.label + ":" + this.targetPromptTokens + ":" + this.requestedNewTokens;
        };
    }

    @JsonCreator(mode = JsonCreator.Mode.DELEGATING)
    public static Scenario parse(String text) {
        String[] parts = text.strip().toLowerCase(Locale.ROOT).split(":");
        try {
            for (Kind kind : Kind.values()) {
                if (!kind.label.equals(parts[0])) continue;
                return switch (kind) {
                    case PREFILL -> {
                        requireParts(parts, 2, text);
                        yield new Scenario(kind, Integer.parseInt(parts[1]), 0);
                    }
                    case FIRST_TOKEN -> {
                        requireParts(parts, 2, text);
                        yield new Scenario(kind, Integer.parseInt(parts[1]), 1);
                    }
                    case PROMPT_TO_N, SUSTAINED_DECODE -> {
                        requireParts(parts, 3, text);
                        yield new Scenario(kind, Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
                    }
                };
            }
        } catch (NumberFormatException invalid) {
            throw new IllegalArgumentException("invalid scenario token count: " + text);
        }
        throw new IllegalArgumentException("unknown scenario kind: " + text);
    }

    private static void requireParts(String[] parts, int count, String text) {
        if (parts.length != count) throw new IllegalArgumentException("invalid scenario: " + text);
    }
}
