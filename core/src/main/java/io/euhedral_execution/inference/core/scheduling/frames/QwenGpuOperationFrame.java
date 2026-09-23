package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenGdnSequenceState;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;

/// Executes one stateful or elementwise GPU instruction using its immutable buffer operands.
public final class QwenGpuOperationFrame extends QwenInstructionFrame {

    public QwenGpuOperationFrame(
            long idHash,
            FrameManager<QwenExecutionContext, QwenGpuOperationFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            ExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        switch (instruction.kind()) {
            case GDN_CONTROL -> runControl(context, instruction);
            case GDN_CONVOLUTION -> runConvolution(context, instruction);
            case GDN_RECURRENCE -> runRecurrence(context, instruction);
            case GDN_GATED_RMS_NORM -> runGatedRmsNorm(context, instruction);
            case RESIDUAL_ADD -> runResidualAdd(context, instruction);
            case SWIGLU -> runSwiGlu(context, instruction);
            default ->
                throw new IllegalArgumentException(
                        "GPU operation frame received unsupported instruction: " + instruction.kind());
        }
    }

    private void runControl(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().gdnControlFp32(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        instruction.weightAddress(0),
                        instruction.weightAddress(1),
                        output(context, instruction, 0),
                        output(context, instruction, 1),
                        context.inputTokenCount(),
                        instruction.outputWidth());
    }

    private void runConvolution(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        int queryKeyWidth = 2 * config.linearNumKeyHeads() * config.linearKeyHeadDim();
        int valueWidth = config.linearNumValueHeads() * config.linearValueHeadDim();
        gpu().gdnConvolutionBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        instruction.weightAddress(0),
                        sequenceState(context).convolutionStateAddress(),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        queryKeyWidth,
                        valueWidth,
                        instruction.outputWidth(),
                        config.linearConvKernelDim());
    }

    private void runRecurrence(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        float outputScale = (float) (1.0 / Math.sqrt(config.linearKeyHeadDim()));
        gpu().gdnRecurrenceBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        input(context, instruction, 2),
                        sequenceState(context).recurrentStateAddress(),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        config.linearNumKeyHeads(),
                        config.linearNumValueHeads(),
                        config.linearKeyHeadDim(),
                        config.linearValueHeadDim(),
                        outputScale);
    }

    private void runGatedRmsNorm(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        QwenConfig config = context.plan().weights().config();
        gpu().gdnGatedRmsNormBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        instruction.weightAddress(0),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        config.linearNumValueHeads(),
                        config.linearValueHeadDim(),
                        (float) config.rmsNormEpsilon());
    }

    private void runResidualAdd(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().residualAddBf16(
                        input(context, instruction, 0),
                        input(context, instruction, 1),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        instruction.outputWidth());
    }

    private void runSwiGlu(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        gpu().swiGluBf16(
                        input(context, instruction, 0),
                        output(context, instruction, 0),
                        context.inputTokenCount(),
                        context.plan().weights().config().intermediateSize());
    }

    private static QwenGdnSequenceState sequenceState(QwenExecutionContext context) {
        Object state = context.sequenceState().recurrentState();
        if (!(state instanceof QwenGdnSequenceState gdnState)) {
            throw new IllegalStateException("sequence is missing its persistent GDN state");
        }
        return gdnState;
    }

    private static long input(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.inputBuffers().get(index));
    }

    private static long output(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction, int index) {
        return context.workspace().address(instruction.outputBuffers().get(index));
    }
}
