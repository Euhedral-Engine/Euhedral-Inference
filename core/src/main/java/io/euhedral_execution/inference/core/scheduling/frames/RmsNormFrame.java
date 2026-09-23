package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;

/// Runs the standalone BF16 RMSNorm instruction.
public final class RmsNormFrame extends QwenInstructionFrame {

    public RmsNormFrame(
            long idHash,
            FrameManager<QwenExecutionContext, RmsNormFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            QwenExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().rmsNormBf16(
                        context.workspace().hiddenStateAddress(),
                        instruction.weight().deviceAddress(),
                        context.workspace().normalizedStateAddress(),
                        context.inputTokenCount(),
                        instruction.outputWidth(),
                        (float) context.plan().weights().config().rmsNormEpsilon());
    }
}
