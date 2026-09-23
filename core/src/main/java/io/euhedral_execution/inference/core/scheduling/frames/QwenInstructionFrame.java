package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;
import java.util.Objects;

/// Base lifecycle for one reusable Qwen instruction frame.
public abstract class QwenInstructionFrame extends AbstractFrame {

    private final QwenExecutionGpu gpu;
    private final QwenWorkGenerator generator;
    private QwenExecutionContext context;
    private final QwenExecutionPlan.Instruction instruction;
    private boolean completed;

    @SuppressWarnings("rawtypes")
    protected QwenInstructionFrame(
            long idHash,
            FrameManager recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            QwenExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, null);
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.generator = Objects.requireNonNull(generator, "generator");
        this.context = Objects.requireNonNull(context, "context");
        this.instruction = Objects.requireNonNull(instruction, "instruction");
        randomizeHash(instruction.id());
    }

    /// Rebind only the per-quantum state; this pool's instruction and weights never change.
    public final void replace(QwenExecutionContext context) {
        this.context = Objects.requireNonNull(context, "context");
        this.completed = false;
        randomizeHash(this.instruction.id());
    }

    protected abstract void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction);

    protected void releaseTemporary(QwenExecutionContext context) {}

    /// Clears an instruction frame that could not be published to Euhedral.
    public final void abandonBeforePublication() {
        this.context = null;
        this.completed = false;
        resetHash();
        recycle();
    }

    protected final QwenExecutionGpu gpu() {
        return this.gpu;
    }

    @Override
    public final void execute() {
        QwenExecutionContext current = this.context;
        if (current.hasFailureOrCancellation()) {
            return;
        }
        perform(current, this.instruction);
        this.gpu.synchronize();
        this.completed = true;
    }

    @Override
    public final void doFinally() {
        finish(this.completed ? null : this.context.failure());
    }

    @Override
    public final void doFinallyWithError(Throwable error) {
        finish(Objects.requireNonNull(error, "error"));
    }

    private void finish(Throwable error) {
        QwenExecutionContext current = this.context;
        QwenExecutionPlan.Instruction completedInstruction = this.instruction;
        try {
            releaseTemporary(current);
        } catch (RuntimeException | Error cleanupFailure) {
            if (error == null) {
                error = cleanupFailure;
            } else {
                error.addSuppressed(cleanupFailure);
            }
        }
        try {
            this.generator.completed(current, completedInstruction, error);
        } finally {
            this.context = null;
            this.completed = false;
            resetHash();
            recycle();
        }
    }
}
