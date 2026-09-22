package io.euhedral_execution.inference.api;

import io.euhedral_execution.inference.core.InferenceEngine;

/**
 * Application entry point for the Euhedral-Inference runtime.
 */
public final class Main {

    private Main() {}

    public static void main(String[] args) {
        InferenceEngine engine = new InferenceEngine();
        System.out.println(engine.describe());
    }
}
