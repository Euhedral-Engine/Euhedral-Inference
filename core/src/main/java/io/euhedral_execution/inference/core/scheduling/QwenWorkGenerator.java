package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/// Turns completed instructions into independently runnable frames using precomputed dependency edges.
final class QwenWorkGenerator {

    private final QwenExecutionPlan plan;
    private final QwenExecutionGpu gpu;
    private final QwenExecutionRunner workQueue;
    private final Consumer<? super QwenExecutionContext> terminalConsumer;

    QwenWorkGenerator(
            QwenExecutionPlan plan,
            QwenExecutionGpu gpu,
            QwenExecutionRunner workQueue,
            Consumer<? super QwenExecutionContext> terminalConsumer) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.workQueue = Objects.requireNonNull(workQueue, "workQueue");
        this.terminalConsumer = Objects.requireNonNull(terminalConsumer, "terminalConsumer");
    }

    void start(QwenExecutionContext context) {
        context.reserveWork();
        publishReserved(context, create(context, plan.instructions().getFirst()));
    }

    void completed(QwenOperationFrame frame, Throwable error) {
        QwenExecutionContext context = frame.context;
        if (error != null) context.fail(error);
        if (!context.hasFailureOrCancellation()) {
            List<QwenOperationFrame> ready = new ArrayList<>();
            try {
                for (int successorId : plan.successors(frame.instruction.id())) {
                    if (context.dependencyCompleted(successorId)) {
                        ready.add(create(context, plan.instructions().get(successorId)));
                    }
                }
                // Reserve before publication: an inline executor may run children immediately.
                for (QwenOperationFrame ignored : ready) context.reserveWork();
                for (QwenOperationFrame successor : ready) publishReserved(context, successor);
            } catch (RuntimeException | Error failure) {
                context.fail(failure);
            }
        }
        if (context.releaseWork()) context.finish(terminalConsumer, gpu);
    }

    private void publishReserved(QwenExecutionContext context, QwenOperationFrame frame) {
        try {
            if (workQueue.offer(frame)) return;
            context.fail(new IllegalStateException("Euhedral work queue rejected a runnable instruction"));
        } catch (RuntimeException | Error failure) {
            context.fail(failure);
        }
        if (context.releaseWork()) context.finish(terminalConsumer, gpu);
    }

    private QwenOperationFrame create(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        return switch (instruction.kind()) {
            case EMBEDDING -> new QwenOperationFrame.EmbeddingFrame(context, instruction, gpu, this);
            case RMS_NORM -> new QwenOperationFrame.RmsNormFrame(context, instruction, gpu, this);
            case Q3_LINEAR -> new QwenOperationFrame.Q3LinearFrame(context, instruction, gpu, this);
        };
    }
}
