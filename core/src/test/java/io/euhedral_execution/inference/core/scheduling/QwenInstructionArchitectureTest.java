package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertFalse;

import io.euhedral_execution.core.frames.PipelineFrame;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class QwenInstructionArchitectureTest {

    @Test
    void executionPlanIsNotAPreconstructedPipeline() {
        assertFalse(Arrays.stream(QwenExecutionPlan.class.getDeclaredFields())
                .anyMatch(field -> PipelineFrame.Builder.class.isAssignableFrom(field.getType())));
    }
}
