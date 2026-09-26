package io.euhedral_execution.inference.core.gpu;

/// Immutable Q3 execution policy. Forced paths are retained for correctness and focused benchmarks.
public enum Q3DispatchMode {
    SCALAR,
    DECODE,
    PREFILL,
    AUTO;

    /// Cold-matrix screens on the model's mixer and MLP shapes favor decode through 8 rows;
    /// the tiled path wins at 9 rows. Kept explicit for other devices and benchmark sweeps.
    public static final int DEFAULT_SMALL_ROW_THRESHOLD = 8;

    public Q3DispatchMode select(int rows, int smallRowThreshold) {
        if (rows <= 0 || smallRowThreshold < 0) throw new IllegalArgumentException("invalid Q3 dispatch dimensions");
        return this == AUTO ? (rows <= smallRowThreshold ? DECODE : PREFILL) : this;
    }
}
