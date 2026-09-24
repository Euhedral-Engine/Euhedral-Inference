package io.euhedral_execution.inference.core;

import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import java.util.Objects;

/// Explicit runtime inputs. CPU IDs are physical/logical processor IDs accepted by Euhedral,
/// not a worker count. The CPU set is defensively copied.
public record InferenceConfig(
        Path artifactPath, Path tokenizerDirectory, Path cudaLibraryPath, BitSet workerCpus, Duration shutdownTimeout) {
    public InferenceConfig {
        Objects.requireNonNull(artifactPath, "artifactPath");
        Objects.requireNonNull(tokenizerDirectory, "tokenizerDirectory");
        Objects.requireNonNull(cudaLibraryPath, "cudaLibraryPath");
        workerCpus = (BitSet) Objects.requireNonNull(workerCpus, "workerCpus").clone();
        if (workerCpus.isEmpty()) throw new IllegalArgumentException("workerCpus must not be empty");
        Objects.requireNonNull(shutdownTimeout, "shutdownTimeout");
        if (shutdownTimeout.isNegative() || shutdownTimeout.isZero())
            throw new IllegalArgumentException("shutdownTimeout must be positive");
        shutdownTimeout.toNanos();
    }

    @Override
    public BitSet workerCpus() {
        return (BitSet) this.workerCpus.clone();
    }
}
