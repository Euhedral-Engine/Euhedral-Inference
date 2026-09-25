package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.AbstractExecutor;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.Objects;

/// The inference execution terminal, cloned once per Euhedral worker.
public final class InferenceGpuExecutor extends AbstractExecutor {
    private final ExecutionGpu gpu;
    private final long worker;
    private InferenceGpuExecutor buddy;
    private boolean closed;

    public InferenceGpuExecutor(ExecutionGpu gpu) {
        this(-1, 0, gpu);
    }

    private InferenceGpuExecutor(int cpu, long worker, ExecutionGpu gpu) {
        super(cpu);
        this.worker = worker;
        this.gpu = Objects.requireNonNull(gpu, "gpu");
    }

    @Override
    public void execute(AbstractFrame frame) {
        gpu.withWorker(worker, frame::execute);
    }

    @Override
    public AbstractExecutor hookOnClone(int cpu) {
        long worker = gpu.openWorker(cpu);
        var cloned = new InferenceGpuExecutor(cpu, worker, gpu);
        if (this.cpu >= 0) this.buddy = cloned;
        return cloned;
    }

    @Override
    public void close() {
        if (closed) return;
        if (buddy != null) buddy.close();
        if (cpu >= 0) gpu.closeWorker(worker);
        closed = true;
    }
}
