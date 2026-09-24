package io.euhedral_execution.inference.core.tokenizer;

/// Classifies ordinary vocabulary entries, control tokens, and the tokenizer EOS token.
public enum TokenKind {
    NORMAL,
    CONTROL,
    EOS
}
