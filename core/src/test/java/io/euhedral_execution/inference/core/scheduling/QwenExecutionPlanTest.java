package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class QwenExecutionPlanTest {
    @Test
    void publishedTopologyAndWeightShapeCannotBeMutatedByCallers() {
        var original = QwenExecutionFixtures.q3("projection", 64, 201);
        var plan =
                new QwenExecutionPlan(QwenExecutionFixtures.weights(), QwenExecutionFixtures.norm(), List.of(original));
        original.shape()[0] = 128;
        plan.instructions().get(2).weight().shape()[0] = 128;
        assertEquals(64, plan.instructions().get(2).weight().shape()[0]);
        assertThrows(
                UnsupportedOperationException.class, () -> plan.instructions().clear());
        assertThrows(
                UnsupportedOperationException.class, () -> plan.successors(1).clear());
        assertEquals(List.of(2), plan.successors(1));
    }
}
