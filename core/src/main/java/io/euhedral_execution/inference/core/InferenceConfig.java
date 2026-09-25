package io.euhedral_execution.inference.core;

import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.Objects;

/// Explicit runtime inputs. [InferenceTuning] is the single owner of the worker processor IDs;
/// [#workerCpus()] reads them from it. CPU IDs are logical processor IDs accepted by Euhedral, not
/// a worker count, and are defensively copied.
public record InferenceConfig(
        Path artifactPath,
        Path tokenizerDirectory,
        Path cudaLibraryPath,
        InferenceTuning tuning,
        Duration shutdownTimeout) {
    public InferenceConfig {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(tokenizerDirectory, "tokenizerDirectory");
        Objects.requireNonNull(cudaLibraryPath, "cudaLibraryPath");
        Objects.requireNonNull(tuning, "tuning");
        Objects.requireNonNull(shutdownTimeout, "shutdownTimeout");
        if (shutdownTimeout.isNegative() || shutdownTimeout.isZero())
            throw new IllegalArgumentException("shutdownTimeout must be positive");
        shutdownTimeout.toNanos();
    }

    /// Uses default tuning for the given worker CPUs.
    public InferenceConfig(
            Path artifactPath,
            Path tokenizerDirectory,
            Path cudaLibraryPath,
            BitSet workerCpus,
            Duration shutdownTimeout) {
        this(
                artifactPath,
                tokenizerDirectory,
                cudaLibraryPath,
                InferenceTuning.defaults(Objects.requireNonNull(workerCpus, "workerCpus")),
                shutdownTimeout);
    }

    /// Returns a copy of the tuning's worker processor IDs.
    public BitSet workerCpus() {
        return this.tuning.workerProcessorIds();
    }
}
