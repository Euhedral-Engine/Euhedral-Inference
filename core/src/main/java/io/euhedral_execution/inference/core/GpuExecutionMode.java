package io.euhedral_execution.inference.core;

/// GPU submission mode. ASYNC_EXPERIMENTAL is opt-in and preserves SYNC as the default.
public enum GpuExecutionMode {
    SYNC,
    ASYNC_EXPERIMENTAL
}
