package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.PipelineFrame;
import io.euhedral_execution.core.ingest.PipelineRunner;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/// Submission-owner wrapper that binds sequence lease release to Euhedral outcome publication.
public final class QwenExecutionRunner {

    private final PipelineRunner<QwenExecutionContext> pipelineRunner;

    QwenExecutionRunner(PipelineRunner<QwenExecutionContext> pipelineRunner) {
        this.pipelineRunner = Objects.requireNonNull(pipelineRunner, "pipelineRunner");
    }

    public CompletableFuture<PipelineFrame.Outcome> submit(QwenExecutionContext context) {
        Objects.requireNonNull(context, "context");
        CompletableFuture<PipelineFrame.Outcome> publishedOutcome = this.pipelineRunner.submit(context);
        CompletableFuture<PipelineFrame.Outcome> finalizedOutcome = new CompletableFuture<>();
        publishedOutcome.whenComplete((outcome, observerFailure) -> {
            try {
                PipelineFrame.Outcome finalized = context.completeOutcome(outcome, observerFailure);
                if (observerFailure == null) {
                    finalizedOutcome.complete(finalized);
                } else {
                    finalizedOutcome.completeExceptionally(observerFailure);
                }
            } catch (Throwable failure) {
                finalizedOutcome.completeExceptionally(failure);
            }
        });
        return finalizedOutcome;
    }

    PipelineRunner<QwenExecutionContext> pipelineRunner() {
        return this.pipelineRunner;
    }

    public void completeGracefully() {
        this.pipelineRunner.completeGracefully();
    }
}
