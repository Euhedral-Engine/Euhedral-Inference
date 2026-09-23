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
        long input = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.inputBuffers().getFirst())
                : context.workspace().hiddenStateAddress();
        long output = context.plan().hasFirstLayer()
                ? context.workspace().address(instruction.outputBuffers().getFirst())
                : context.workspace().normalizedStateAddress();
        float epsilon = (float) context.plan().weights().config().rmsNormEpsilon();
        if (instruction.kind() == QwenExecutionPlan.Kind.RMS_NORM_UNIT_OFFSET) {
            gpu().rmsNormUnitOffsetBf16(
                            input,
                            instruction.weightAddress(),
                            output,
                            context.inputTokenCount(),
                            instruction.outputWidth(),
                            epsilon);
        } else {
            gpu().rmsNormBf16(
                            input,
                            instruction.weightAddress(),
                            output,
                            context.inputTokenCount(),
                            instruction.outputWidth(),
                            epsilon);
        }
    }
}
