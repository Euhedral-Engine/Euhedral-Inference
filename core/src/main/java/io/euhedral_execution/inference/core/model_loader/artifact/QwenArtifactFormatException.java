package io.euhedral_execution.inference.core.model_loader.artifact;

import java.io.IOException;

public final class QwenArtifactFormatException extends IOException {

    private static final long serialVersionUID = 1L;

    public QwenArtifactFormatException(String message) {
        super(message);
    }
}
