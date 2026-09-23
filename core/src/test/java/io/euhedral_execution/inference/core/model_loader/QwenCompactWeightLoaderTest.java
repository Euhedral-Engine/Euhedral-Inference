package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactMtpAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class QwenCompactWeightLoaderTest {

    @Test
    void assemblesFusedRuntimeObjectsWithoutSplittingHandles() throws Exception {
        QwenConfig config = config();
        Map<String, TensorHandle> handles = new LinkedHashMap<>();
        handles.put("text/token_embedding", handle("text/token_embedding"));
        handles.put("text/final_norm", handle("text/final_norm"));
        handles.put("text/output_head", handle("text/output_head"));
        for (int index = 0; index < 64; index++) {
            String prefix = "text/layers/" + index;
            handles.put(prefix + "/input_norm", handle(prefix + "/input_norm"));
            handles.put(prefix + "/post_attention_norm", handle(prefix + "/post_attention_norm"));
            handles.put(prefix + "/mlp/gate_up", handle(prefix + "/mlp/gate_up"));
            handles.put(prefix + "/mlp/down", handle(prefix + "/mlp/down"));
            if (config.layerTypes()[index] == QwenLayerType.FULL_ATTENTION) {
                for (String suffix : new String[] {
                    "/attention/query_key",
                    "/attention/gate_value",
                    "/attention/query_norm",
                    "/attention/key_norm",
                    "/attention/output"
                }) {
                    handles.put(prefix + suffix, handle(prefix + suffix));
                }
            } else {
                for (String suffix : new String[] {
                    "/gdn/a_log",
                    "/gdn/dt_bias",
                    "/gdn/convolution",
                    "/gdn/a_projection",
                    "/gdn/b_projection",
                    "/gdn/query_key",
                    "/gdn/value_z",
                    "/gdn/norm",
                    "/gdn/output"
                }) {
                    handles.put(prefix + suffix, handle(prefix + suffix));
                }
            }
        }
        handles.put("mtp/input_projection", handle("mtp/input_projection"));
        handles.put("mtp/embedding_norm", handle("mtp/embedding_norm"));
        handles.put("mtp/hidden_norm", handle("mtp/hidden_norm"));
        handles.put("mtp/final_norm", handle("mtp/final_norm"));
        handles.put("mtp/layer/input_norm", handle("mtp/layer/input_norm"));
        handles.put("mtp/layer/post_attention_norm", handle("mtp/layer/post_attention_norm"));
        for (String suffix : new String[] {
            "/attention/query_key_gate_value",
            "/attention/query_norm",
            "/attention/key_norm",
            "/attention/output",
            "/mlp/gate_up",
            "/mlp/down"
        }) {
            handles.put("mtp/layer" + suffix, handle("mtp/layer" + suffix));
        }
        handles.put("text/draft_head", handle("text/draft_head"));
        handles.put("text/draft_head_token_ids", handle("text/draft_head_token_ids"));
        handles.put("vision/sentinel", handle("vision/sentinel"));

        QwenWeights weights = QwenCompactWeightLoader.assemble(config, handles);

        QwenLayerWeights gdnLayer = weights.layers()[0];
        assertInstanceOf(QwenCompactGatedDeltaNetWeights.class, gdnLayer.mixer());
        assertInstanceOf(QwenCompactDenseFfnWeights.class, gdnLayer.ffn());
        assertInstanceOf(QwenCompactAttentionWeights.class, weights.layers()[3].mixer());
        assertEquals(
                "text/layers/3/attention/query_key",
                ((QwenCompactAttentionWeights) weights.layers()[3].mixer())
                        .queryKey()
                        .name());
        assertInstanceOf(
                QwenCompactMtpAttentionWeights.class, weights.mtp().layer().mixer());
        assertEquals(handles.size(), weights.runtimeObjects().size());
        assertThrows(
                UnsupportedOperationException.class,
                () -> weights.runtimeObjects().put("unexpected", handle("unexpected")));
    }

    private static TensorHandle handle(String name) {
        return new TensorHandle(name, new long[] {1}, TensorDataType.BF16, WeightFormat.BF16, 1L, 2L);
    }

    private static QwenConfig config() {
        QwenLayerType[] types = new QwenLayerType[64];
        for (int index = 0; index < types.length; index++) {
            types[index] = index % 4 == 3 ? QwenLayerType.FULL_ATTENTION : QwenLayerType.GATED_DELTA_NET;
        }
        return new QwenConfig(
                248320,
                5120,
                64,
                24,
                4,
                256,
                17408,
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
                types,
                0,
                0,
                0,
                0,
                false,
                true,
                1);
    }
}
