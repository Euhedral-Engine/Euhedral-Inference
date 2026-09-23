package io.euhedral_execution.inference.core.model_loader;

import java.io.IOException;

/// Signals that a Qwen artifact cannot be assembled into the declared weight structure.
public final class QwenWeightLoadException extends IOException {

    public QwenWeightLoadException(String message) {
        super(message);
    }

    public QwenWeightLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}
