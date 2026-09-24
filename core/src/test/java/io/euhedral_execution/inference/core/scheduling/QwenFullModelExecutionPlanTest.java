package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import org.junit.jupiter.api.Test;

class QwenFullModelExecutionPlanTest {

    private static final int HIDDEN = 5120;
    private static final int INTERMEDIATE = 17408;
    private static final int VOCABULARY = 248320;
    private static long nextAddress = 10_000;

    @Test
    void buildsOneWeightBoundGraphForAllDeclaredLayersAndFinalLogits() {
        QwenWeights weights = fullModelWeights();

        QwenExecutionPlan plan = new QwenExecutionPlan(weights);

        assertEquals(979, plan.instructions().size());
        assertEquals(
                QwenExecutionPlan.Kind.EMBEDDING, plan.instructions().getFirst().kind());
        assertEquals(
                48,
                plan.instructions().stream()
                        .filter(instruction -> instruction.kind() == QwenExecutionPlan.Kind.GDN_RECURRENCE)
                        .count());
        assertEquals(
                193,
                plan.instructions().stream()
                        .filter(instruction -> instruction.kind() == QwenExecutionPlan.Kind.Q3_LINEAR)
                        .count());
        assertTrue(plan.instructions().stream()
                .flatMap(instruction -> instruction.weights().stream())
                .anyMatch(weight -> weight.name().equals("text/layers/63/mlp/down")));
        assertEquals("text/output_head", plan.instructions().getLast().weightName(0));
        assertEquals(VOCABULARY, plan.instructions().getLast().outputWidth());
    }

    @Test
    void detachesGpuLogitsFromReusableQuantumWorkspace() {
        QwenExecutionPlan plan = new QwenExecutionPlan(fullModelWeights());
        QwenExecutionFixtures.RecordingGpu gpu = new QwenExecutionFixtures.RecordingGpu();
        QwenExecutionWorkspace workspace = new QwenExecutionWorkspace(gpu, 2, plan);
        workspace.allocateBuffers();
        long logits = workspace.address(QwenExecutionPlan.Buffer.LOGITS);

        assertEquals((long) 2 * VOCABULARY * Short.BYTES, workspace.bufferByteSize(QwenExecutionPlan.Buffer.LOGITS));
        assertEquals(logits, workspace.detachAddress(QwenExecutionPlan.Buffer.LOGITS));
        workspace.close();

        assertTrue(workspace.isClosed());
        assertEquals(false, gpu.frees.contains(logits));
        gpu.free(logits);
        assertTrue(gpu.frees.contains(logits));
    }

    @Test
    void stagedPrefixEndsAtSelectedMixedLayerWithoutClaimingVocabularyOutput() {
        QwenExecutionPlan plan = QwenExecutionPlan.prefix(fullModelWeights(), 4);

        assertEquals(
                4,
                plan.instructions().stream()
                                .mapToInt(QwenExecutionPlan.Instruction::layerIndex)
                                .filter(layer -> layer >= 0)
                                .max()
                                .orElseThrow()
                        + 1);
        assertEquals(3, plan.instructions().getLast().layerIndex());
        assertEquals(
                QwenExecutionPlan.Kind.RESIDUAL_ADD,
                plan.instructions().getLast().kind());
        assertTrue(plan.instructions().stream()
                .noneMatch(instruction -> instruction.outputBuffers().contains(QwenExecutionPlan.Buffer.LOGITS)));
        assertTrue(plan.bufferSpecs().stream().noneMatch(spec -> spec.buffer() == QwenExecutionPlan.Buffer.LOGITS));
    }

    private static QwenWeights fullModelWeights() {
        QwenLayerType[] layerTypes = new QwenLayerType[64];
        QwenLayerWeights[] layers = new QwenLayerWeights[layerTypes.length];
        for (int layerIndex = 0; layerIndex < layerTypes.length; layerIndex++) {
            boolean fullAttention = layerIndex % 4 == 3;
            layerTypes[layerIndex] = fullAttention ? QwenLayerType.FULL_ATTENTION : QwenLayerType.GATED_DELTA_NET;
            String prefix = "text/layers/" + layerIndex;
            var mixer = fullAttention ? attentionWeights(prefix) : gdnWeights(prefix);
            layers[layerIndex] = new QwenLayerWeights(
                    layerIndex,
                    direct(prefix + "/input_norm", WeightFormat.BF16, HIDDEN),
                    direct(prefix + "/post_attention_norm", WeightFormat.BF16, HIDDEN),
                    mixer,
                    new QwenCompactDenseFfnWeights(
                            quantized(prefix + "/mlp/gate_up", 2 * INTERMEDIATE, HIDDEN, WeightFormat.Q3_G64_FP16),
                            quantized(prefix + "/mlp/down", HIDDEN, INTERMEDIATE, WeightFormat.Q3_G64_FP16)));
        }
        QwenConfig config = new QwenConfig(
                VOCABULARY,
                HIDDEN,
                layerTypes.length,
                24,
                4,
                256,
                INTERMEDIATE,
                16,
                48,
                128,
                128,
                4,
                1.0e-6,
                10_000_000.0,
                0.25,
                262144,
                "silu",
                layerTypes,
                0,
                0,
                0,
                0,
                false,
                true,
                1);
        return new QwenWeights(
                config,
                quantized("text/token_embedding", VOCABULARY, HIDDEN, WeightFormat.Q3_G64_FP16),
                layers,
                direct("text/final_norm", WeightFormat.BF16, HIDDEN),
                quantized("text/output_head", VOCABULARY, HIDDEN, WeightFormat.Q3_G64_FP16),
                null);
    }

    private static QwenCompactAttentionWeights attentionWeights(String prefix) {
        return new QwenCompactAttentionWeights(
                quantized(prefix + "/attention/query_key", 7168, HIDDEN, WeightFormat.Q4_G64_FP16),
                quantized(prefix + "/attention/gate_value", 7168, HIDDEN, WeightFormat.Q5_G64_FP16),
                direct(prefix + "/attention/query_norm", WeightFormat.BF16, 256),
                direct(prefix + "/attention/key_norm", WeightFormat.BF16, 256),
                quantized(prefix + "/attention/output", HIDDEN, 6144, WeightFormat.Q3_G64_FP16));
    }

    private static QwenCompactGatedDeltaNetWeights gdnWeights(String prefix) {
        return new QwenCompactGatedDeltaNetWeights(
                direct(prefix + "/gdn/a_log", WeightFormat.FP32, 48),
                direct(prefix + "/gdn/dt_bias", WeightFormat.FP32, 48),
                direct(prefix + "/gdn/convolution", WeightFormat.BF16, 4, 10240),
                direct(prefix + "/gdn/a_projection", WeightFormat.BF16, 48, HIDDEN),
                direct(prefix + "/gdn/b_projection", WeightFormat.BF16, 48, HIDDEN),
                quantized(prefix + "/gdn/query_key", 4096, HIDDEN, WeightFormat.Q4_G64_FP16),
                quantized(prefix + "/gdn/value_z", 12288, HIDDEN, WeightFormat.Q5_G64_FP16),
                direct(prefix + "/gdn/norm", WeightFormat.BF16, 128),
                quantized(prefix + "/gdn/output", HIDDEN, 6144, WeightFormat.Q3_G64_FP16));
    }

    private static TensorHandle direct(String name, WeightFormat format, int... dimensions) {
        long[] shape = new long[dimensions.length];
        long elements = 1;
        for (int index = 0; index < dimensions.length; index++) {
            shape[index] = dimensions[index];
            elements = Math.multiplyExact(elements, dimensions[index]);
        }
        int bytesPerElement = format == WeightFormat.FP32 ? Float.BYTES : Short.BYTES;
        return new TensorHandle(
                name,
                shape,
                TensorDataType.BF16,
                format,
                WeightLayout.CONTIGUOUS_LE_V1,
                nextAddress++,
                Math.multiplyExact(elements, bytesPerElement));
    }

    private static TensorHandle quantized(String name, int rows, int columns, WeightFormat format) {
        long[] shape = {rows, columns};
        return new TensorHandle(
                name,
                shape,
                TensorDataType.BF16,
                format,
                WeightLayout.ROW_SPLIT_K128_V1,
                nextAddress++,
                CompactTensorLayout.expectedByteSize(
                        shape, TensorDataType.BF16, format, WeightLayout.ROW_SPLIT_K128_V1));
    }
}
