package io.euhedral_execution.inference.api.engine;

/// The engine cannot admit or finish work because it is closing, closed, or saturated.
public final class InferenceUnavailableException extends RuntimeException {

    public InferenceUnavailableException(String message) {
        super(message);
    }
}
