package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;

/// Runs one independent Q3 projection over the normalized BF16 input.
public final class LinearFrame extends QwenInstructionFrame {

    public LinearFrame(
            long idHash,
            FrameManager<QwenExecutionContext, LinearFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            QwenExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        var weight = instruction.weight();
        gpu().linearQ3Bf16(
                        context.workspace().normalizedStateAddress(),
                        weight.deviceAddress(),
                        context.workspace().projectionAddress(instruction.id() - 2),
                        context.inputTokenCount(),
                        context.plan().weights().config().hiddenSize(),
                        instruction.outputWidth(),
                        weight.byteSize());
    }
}
