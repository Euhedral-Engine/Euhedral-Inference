package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenWorkGenerator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// Runs the embedding lookup instruction and owns its temporary token-ID upload until finalization.
public final class EmbeddingFrame extends QwenInstructionFrame {
    private ExecutionGpu.UploadBuffer pendingUpload;

    public EmbeddingFrame(
            long idHash,
            FrameManager<QwenExecutionContext, EmbeddingFrame> recycler,
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            ExecutionGpu gpu,
            QwenWorkGenerator generator) {
        super(idHash, recycler, context, instruction, gpu, generator);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        int[] ids = context.inputTokenIds();
        long bytes = (long) ids.length * Integer.BYTES;
        if (gpu().asynchronous()) {
            this.pendingUpload = gpu().allocateUploadBuffer(bytes);
            uploadAndEmbed(context, instruction, ids, bytes, this.pendingUpload.segment(), true);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(bytes, Integer.BYTES);
            uploadAndEmbed(context, instruction, ids, bytes, host, false);
        }
    }

    private void uploadAndEmbed(
            QwenExecutionContext context,
            QwenExecutionPlan.Instruction instruction,
            int[] ids,
            long bytes,
            MemorySegment host,
            boolean deferred) {
        for (int index = 0; index < ids.length; index++) {
            host.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, ids[index]);
        }
        long tokenBuffer = context.allocateTemporaryTokenIds(gpu(), bytes);
        if (deferred) gpu().copyUploadToDevice(tokenBuffer, this.pendingUpload);
        else gpu().copyHostToDevice(tokenBuffer, host, bytes);
        gpu().embedQ3(
                        tokenBuffer,
                        instruction.weightAddress(),
                        instruction.weightByteSize(),
                        context.workspace().hiddenStateAddress(),
                        ids.length,
                        context.plan().weights().config().vocabSize(),
                        instruction.outputWidth());
    }

    @Override
    protected void releaseTemporary(QwenExecutionContext context) {
        try {
            context.releaseTemporaryTokenIds(gpu());
        } finally {
            ExecutionGpu.UploadBuffer upload = this.pendingUpload;
            this.pendingUpload = null;
            // A poisoned GPU cannot prove that DMA has stopped reading pinned host memory.
            if (upload != null && gpu().completionProven()) upload.close();
        }
    }
}
