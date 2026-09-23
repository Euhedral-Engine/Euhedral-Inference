package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// One executable GPU instruction. Frame-local state has one worker owner and needs no lock.
abstract class QwenOperationFrame extends AbstractFrame {

    final QwenExecutionContext context;
    final QwenExecutionPlan.Instruction instruction;
    final QwenExecutionGpu gpu;
    final QwenWorkGenerator generator;
    private boolean completed;

    QwenOperationFrame(
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            QwenExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(context.sequenceState().sequenceId());
        this.context = context;
        this.instruction = instruction;
        this.gpu = gpu;
        this.generator = generator;
        randomizeHash(instruction.id());
    }

    @Override
    public final void execute() {
        if (context.hasFailureOrCancellation()) {
            return;
        }
        perform();
        gpu.synchronize();
        completed = true;
    }

    abstract void perform();

    @Override
    public final void doFinally() {
        finish(completed ? null : context.failure());
    }

    @Override
    public final void doFinallyWithError(Throwable error) {
        finish(error);
    }

    private void finish(Throwable error) {
        try {
            releaseTemporary();
        } catch (RuntimeException | Error cleanupFailure) {
            if (error == null) {
                error = cleanupFailure;
            } else {
                error.addSuppressed(cleanupFailure);
            }
        }
        generator.completed(this, error);
    }

    void releaseTemporary() {}

    static final class EmbeddingFrame extends QwenOperationFrame {
        EmbeddingFrame(
                QwenExecutionContext context,
                QwenExecutionPlan.Instruction instruction,
                QwenExecutionGpu gpu,
                QwenWorkGenerator generator) {
            super(context, instruction, gpu, generator);
        }

        @Override
        void perform() {
            int[] ids = context.inputTokenIds();
            long bytes = (long) ids.length * Integer.BYTES;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment host = arena.allocate(bytes, Integer.BYTES);
                for (int index = 0; index < ids.length; index++) {
                    host.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, ids[index]);
                }
                long tokenBuffer = context.allocateTemporaryTokenIds(gpu, bytes);
                gpu.copyHostToDevice(tokenBuffer, host, bytes);
                TensorHandle weight = instruction.weight();
                gpu.embedQ3(
                        tokenBuffer,
                        weight.deviceAddress(),
                        weight.byteSize(),
                        context.workspace().hiddenStateAddress(),
                        ids.length,
                        context.plan().weights().config().vocabSize(),
                        instruction.outputWidth());
            }
        }

        @Override
        void releaseTemporary() {
            context.releaseTemporaryTokenIds(gpu);
        }
    }

    static final class RmsNormFrame extends QwenOperationFrame {
        RmsNormFrame(
                QwenExecutionContext context,
                QwenExecutionPlan.Instruction instruction,
                QwenExecutionGpu gpu,
                QwenWorkGenerator generator) {
            super(context, instruction, gpu, generator);
        }

        @Override
        void perform() {
            gpu.rmsNormBf16(
                    context.workspace().hiddenStateAddress(),
                    instruction.weight().deviceAddress(),
                    context.workspace().normalizedStateAddress(),
                    context.inputTokenCount(),
                    instruction.outputWidth(),
                    (float) context.plan().weights().config().rmsNormEpsilon());
        }
    }

    static final class Q3LinearFrame extends QwenOperationFrame {
        Q3LinearFrame(
                QwenExecutionContext context,
                QwenExecutionPlan.Instruction instruction,
                QwenExecutionGpu gpu,
                QwenWorkGenerator generator) {
            super(context, instruction, gpu, generator);
        }

        @Override
        void perform() {
            TensorHandle weight = instruction.weight();
            gpu.linearQ3Bf16(
                    context.workspace().normalizedStateAddress(),
                    weight.deviceAddress(),
                    context.workspace().projectionAddress(instruction.id() - 2),
                    context.inputTokenCount(),
                    context.plan().weights().config().hiddenSize(),
                    instruction.outputWidth(),
                    weight.byteSize());
        }
    }
}
