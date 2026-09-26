package io.euhedral_execution.inference.core.scheduling;

/// Vocabulary output needed by a quantum's caller; transformer state updates are unaffected.
public enum QwenLogitsRequirement {
    NONE,
    LAST_TOKEN,
    ALL_TOKENS;

    public int outputRows(int inputRows) {
        if (inputRows <= 0) throw new IllegalArgumentException("inputRows must be positive");
        return switch (this) {
            case NONE -> 0;
            case LAST_TOKEN -> 1;
            case ALL_TOKENS -> inputRows;
        };
    }
}
