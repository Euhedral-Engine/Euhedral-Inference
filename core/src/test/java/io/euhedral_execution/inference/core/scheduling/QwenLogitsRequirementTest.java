package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.inference.core.scheduling.frames.LinearFrame;
import io.euhedral_execution.inference.core.scheduling.frames.RmsNormFrame;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class QwenLogitsRequirementTest {
    @ParameterizedTest
    @EnumSource(QwenLogitsRequirement.class)
    void outputRequirementControlsAllocationNormalizationAndProjection(QwenLogitsRequirement requirement) {
        // A prime vocabulary size distinguishes logits bytes from power-of-two state buffers.
        var plan = new QwenExecutionPlan(QwenExecutionFixtures.statefulCompactWeights(1009));
        var gpu = new OutputGpu();
        var sequence = new QwenSequenceState(71);
        var context = new QwenExecutionContext(
                plan, sequence, QwenExecutionContext.ExecutionKind.PREFILL, 0, new int[] {1, 2, 3}, requirement);
        var runner = new QwenExecutionRunner(plan, gpu);
        var generator = new QwenWorkGenerator(plan, gpu, runner, ignored -> {});
        context.begin(gpu);
        try {
            int rows = requirement.outputRows(3);
            assertEquals(requirement, context.logitsRequirement());
            var workspace = context.workspace();
            var norm = plan.instructions().get(plan.instructions().size() - 2);
            var projection = plan.instructions().getLast();
            long finalHidden = workspace.address(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE);
            assertEquals(rows != 0, workspace.hasBuffer(QwenExecutionPlan.Buffer.LOGITS));
            assertEquals(rows != 0, workspace.hasBuffer(QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
            if (rows != 0) {
                assertEquals(
                        (long) rows * 1009 * Short.BYTES, workspace.bufferByteSize(QwenExecutionPlan.Buffer.LOGITS));
            }
            new RmsNormFrame(1, new FrameManager<>(8, 1), context, norm, gpu, generator).execute();
            new LinearFrame(2, new FrameManager<>(8, 1), context, projection, gpu, generator).execute();
            assertEquals(rows == 0 ? List.of() : List.of(rows), gpu.normRows);
            assertEquals(rows == 0 ? List.of() : List.of(rows), gpu.projectionRows);
            if (rows != 0) {
                assertEquals(
                        finalHidden
                                + (requirement == QwenLogitsRequirement.LAST_TOKEN
                                        ? 2L * norm.inputWidth() * Short.BYTES
                                        : 0),
                        gpu.normInput);
                assertEquals(workspace.address(QwenExecutionPlan.Buffer.FINAL_NORMALIZED), gpu.projectionInput);
            }
            context.finish(null, gpu);
            assertEquals(
                    requirement == QwenLogitsRequirement.LAST_TOKEN ? 1 : 0,
                    java.util.Collections.frequency(gpu.allocationSizes, 1009L * Short.BYTES));
            assertEquals(
                    requirement == QwenLogitsRequirement.ALL_TOKENS ? 1 : 0,
                    java.util.Collections.frequency(gpu.allocationSizes, 3L * 1009 * Short.BYTES));
            assertEquals(
                    QwenExecutionContext.Status.SUCCESS,
                    context.outcome().join().status());
            assertEquals(3, sequence.currentTokenPosition());
            assertEquals(rows != 0, context.logitsOutput().isPresent());
            context.logitsOutput().ifPresent(logits -> {
                assertEquals(rows, logits.tokenCount());
                logits.close();
            });
        } finally {
            context.workspace().close();
            sequence.complete();
            runner.completeGracefully();
        }
    }

    private static class OutputGpu extends QwenExecutionFixtures.RecordingGpu {
        final List<Integer> normRows = new ArrayList<>();
        final List<Integer> projectionRows = new ArrayList<>();
        final List<Long> allocationSizes = new ArrayList<>();
        long normInput;
        long projectionInput;

        @Override
        public long allocate(long bytes) {
            allocationSizes.add(bytes);
            return super.allocate(bytes);
        }

        @Override
        public void zeroDeviceMemory(long address, long bytes) {}

        @Override
        public void rmsNormUnitOffsetBf16(long input, long weight, long output, int rows, int width, float epsilon) {
            normRows.add(rows);
            normInput = input;
        }

        @Override
        public void linearQ3Bf16(
                long input, long weights, long output, int rows, int inFeatures, int outFeatures, long bytes) {
            projectionRows.add(rows);
            projectionInput = input;
        }
    }
}
