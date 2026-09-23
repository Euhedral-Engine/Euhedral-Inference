package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactHeader;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMoeWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMtpWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class QwenWeightLoaderTest {

    @TempDir
    Path tempDirectory;

    @Test
    void compactDispatchRequiresCompactArtifactVersion() {
        QwenConfig config = config(0, QwenLayerType.FULL_ATTENTION, 0, 0);
        TensorDescriptor descriptor =
                new TensorDescriptor("text/not_compact", new long[] {1}, TensorDataType.FP32, WeightFormat.FP32, 0, 0);
        QwenArtifactHeader rawHeader = new QwenArtifactHeader(
                QwenArtifactHeader.MAGIC,
                QwenArtifactHeader.VERSION,
                QwenArtifactHeader.BYTE_SIZE,
                0,
                QwenArtifactHeader.BYTE_SIZE,
                1,
                QwenArtifactHeader.BYTE_SIZE);
        QwenArtifactHeader compactHeader = new QwenArtifactHeader(
                QwenArtifactHeader.MAGIC,
                QwenArtifactHeader.COMPACT_VERSION,
                QwenArtifactHeader.BYTE_SIZE,
                0,
                QwenArtifactHeader.BYTE_SIZE,
                1,
                QwenArtifactHeader.BYTE_SIZE);

        assertFalse(QwenWeightLoader.isCompactArtifact(
                new QwenArtifact(rawHeader, config, new TensorDescriptor[] {descriptor})));
        assertTrue(QwenWeightLoader.isCompactArtifact(
                new QwenArtifact(compactHeader, config, new TensorDescriptor[] {descriptor})));
    }

    @Test
    void assemblesSuccessfulDenseModel() throws Exception {
        QwenConfig config = config(1, QwenLayerType.FULL_ATTENTION, 0, 0);
        String[] names = denseLayerNames("model.layers.0", false);
        QwenArtifact artifact = artifact(
                config,
                concat(new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}, names));
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeights weights = QwenWeightLoader.load(emptyArtifactFile(), artifact, gpu);

        assertEquals(config, weights.config());
        assertEquals("model.embed_tokens.weight", weights.tokenEmbedding().name());
        assertEquals("model.norm.weight", weights.finalNorm().name());
        assertEquals("lm_head.weight", weights.lmHead().name());
        assertEquals(1, weights.layers().length);
        assertInstanceOf(QwenAttentionWeights.class, weights.layers()[0].mixer());
        assertInstanceOf(QwenDenseFfnWeights.class, weights.layers()[0].ffn());
        assertEquals(null, weights.mtp());
        assertEquals(names.length + 3, gpu.allocateCalls);
        assertTrue(gpu.freedAddresses.isEmpty());
    }

    @Test
    void assemblesMixedAttentionAndGdnLayers() throws Exception {
        QwenConfig config = config(2, QwenLayerType.FULL_ATTENTION, 0, 0, QwenLayerType.GATED_DELTA_NET);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                denseLayerNames("model.layers.0", false),
                gdnLayerNames("model.layers.1"));
        QwenWeights weights = QwenWeightLoader.load(emptyArtifactFile(), artifact(config, names), new FakeGpuMemory());

        assertInstanceOf(QwenAttentionWeights.class, weights.layers()[0].mixer());
        assertInstanceOf(QwenGatedDeltaNetWeights.class, weights.layers()[1].mixer());
        assertEquals(
                "model.layers.1.linear_attn.conv1d.weight",
                weights.layers()[1].mixer() instanceof QwenGatedDeltaNetWeights gdn
                        ? gdn.convid().name()
                        : "");
    }

    @Test
    void assemblesMoeExpertsAndSharedExpert() throws Exception {
        QwenConfig config = config(1, new QwenLayerType[] {QwenLayerType.FULL_ATTENTION}, 2, 1, 2, 4, 0);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                attentionNames("model.layers.0"),
                new String[] {
                    "model.layers.0.input_layernorm.weight",
                    "model.layers.0.post_attention_layernorm.weight",
                    "model.layers.0.mlp.gate.weight",
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    "model.layers.0.mlp.experts.1.gate_proj.weight",
                    "model.layers.0.mlp.experts.1.up_proj.weight",
                    "model.layers.0.mlp.experts.1.down_proj.weight",
                    "model.layers.0.mlp.shared_expert.gate_proj.weight",
                    "model.layers.0.mlp.shared_expert.up_proj.weight",
                    "model.layers.0.mlp.shared_expert.down_proj.weight",
                    "model.layers.0.mlp.shared_expert_gate.weight"
                });

        QwenLayerWeights layer = QwenWeightLoader.load(
                        emptyArtifactFile(), artifact(config, names), new FakeGpuMemory())
                .layers()[0];

        QwenMoeWeights moe = assertInstanceOf(QwenMoeWeights.class, layer.ffn());
        assertEquals("model.layers.0.mlp.gate.weight", moe.router().name());
        assertEquals(2, moe.experts().length);
        assertEquals(
                "model.layers.0.mlp.experts.1.down_proj.weight",
                moe.experts()[1].downProj().name());
        assertEquals(
                "model.layers.0.mlp.shared_expert_gate.weight",
                moe.sharedExpertGate().name());
    }

    @Test
    void assemblesMtpWhenConfigured() throws Exception {
        QwenConfig config = config(1, QwenLayerType.FULL_ATTENTION, 0, 0, 1);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                denseLayerNames("model.layers.0", false),
                new String[] {
                    "mtp.pre_fc_norm_embedding.weight",
                    "mtp.pre_fc_norm_hidden.weight",
                    "mtp.fc.weight",
                    "mtp.norm.weight"
                },
                denseLayerNames("mtp.layers.0", false));

        QwenMtpWeights mtp = QwenWeightLoader.load(emptyArtifactFile(), artifact(config, names), new FakeGpuMemory())
                .mtp();

        assertEquals("mtp.pre_fc_norm_embedding.weight", mtp.embeddingNorm().name());
        assertEquals("mtp.pre_fc_norm_hidden.weight", mtp.hiddenNorm().name());
        assertEquals("mtp.fc.weight", mtp.projection().name());
        assertEquals("mtp.norm.weight", mtp.finalNorm().name());
        assertInstanceOf(QwenAttentionWeights.class, mtp.layer().mixer());
    }

    @Test
    void leavesMtpAbsentWhenNotConfigured() throws Exception {
        QwenConfig config = config(1, QwenLayerType.FULL_ATTENTION, 0, 0);
        QwenWeights weights = QwenWeightLoader.load(
                emptyArtifactFile(),
                artifact(
                        config,
                        concat(
                                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                                denseLayerNames("model.layers.0", false))),
                new FakeGpuMemory());

        assertEquals(null, weights.mtp());
    }

    @Test
    void rejectsMissingRequiredTensorBeforeGpuLoad() throws Exception {
        QwenConfig config = config(1, QwenLayerType.FULL_ATTENTION, 0, 0);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight"},
                denseLayerNames("model.layers.0", false));
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeightLoadException failure = assertThrows(
                QwenWeightLoadException.class,
                () -> QwenWeightLoader.load(emptyArtifactFile(), artifact(config, names), gpu));

        assertTrue(failure.getMessage().contains("lm_head.weight"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsDuplicateTensorNameBeforeGpuLoad() throws Exception {
        QwenConfig config = config(0, new QwenLayerType[0], 0, 0);
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeightLoadException failure = assertThrows(
                QwenWeightLoadException.class,
                () -> QwenWeightLoader.load(
                        emptyArtifactFile(),
                        artifact(
                                config,
                                "model.embed_tokens.weight",
                                "model.embed_tokens.weight",
                                "model.norm.weight",
                                "lm_head.weight"),
                        gpu));

        assertTrue(failure.getMessage().contains("duplicate tensor name"));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void rejectsConfigTensorMismatch() throws Exception {
        QwenConfig config = config(1, QwenLayerType.GATED_DELTA_NET, 0, 0);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                denseLayerNames("model.layers.0", false));
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeightLoadException failure = assertThrows(
                QwenWeightLoadException.class,
                () -> QwenWeightLoader.load(emptyArtifactFile(), artifact(config, names), gpu));

        assertTrue(failure.getMessage().contains("linear_attn."));
        assertEquals(0, gpu.allocateCalls);
    }

    @Test
    void freesEveryAllocationWhenTensorLoadFailsPartWayThrough() throws Exception {
        QwenConfig config = config(1, QwenLayerType.FULL_ATTENTION, 0, 0);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                denseLayerNames("model.layers.0", false));
        FakeGpuMemory gpu = new FakeGpuMemory();
        gpu.failAtAllocation = 4;

        assertThrows(
                RuntimeException.class, () -> QwenWeightLoader.load(emptyArtifactFile(), artifact(config, names), gpu));

        assertEquals(4, gpu.allocateCalls);
        assertEquals(new HashSet<>(gpu.allocatedAddresses), new HashSet<>(gpu.freedAddresses));
    }

    @Test
    void freesEveryAllocationWhenAssemblyRejectsUnexpectedTensor() throws Exception {
        QwenConfig config = config(1, QwenLayerType.FULL_ATTENTION, 0, 0);
        String[] names = concat(
                new String[] {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
                denseLayerNames("model.layers.0", false),
                new String[] {"unexpected.tensor"});
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeightLoadException failure = assertThrows(
                QwenWeightLoadException.class,
                () -> QwenWeightLoader.load(emptyArtifactFile(), artifact(config, names), gpu));

        assertTrue(failure.getMessage().contains("unexpected.tensor"));
        assertEquals(new HashSet<>(gpu.allocatedAddresses), new HashSet<>(gpu.freedAddresses));
    }

    @Test
    void successfulAssemblyTransfersAllocationOwnershipToWeights() throws Exception {
        QwenConfig config = config(0, new QwenLayerType[0], 0, 0);
        FakeGpuMemory gpu = new FakeGpuMemory();

        QwenWeights weights = QwenWeightLoader.load(
                emptyArtifactFile(),
                artifact(config, "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"),
                gpu);

        assertTrue(gpu.freedAddresses.isEmpty());
        assertEquals(
                gpu.addressFor("model.embed_tokens.weight"),
                weights.tokenEmbedding().deviceAddress());
        assertEquals(gpu.addressFor("model.norm.weight"), weights.finalNorm().deviceAddress());
        assertEquals(gpu.addressFor("lm_head.weight"), weights.lmHead().deviceAddress());
    }

    private Path emptyArtifactFile() throws Exception {
        Path path = tempDirectory.resolve("weights-" + Files.list(tempDirectory).count() + ".edrl");
        Files.write(path, new byte[0]);
        return path;
    }

    private static QwenArtifact artifact(QwenConfig config, String... names) {
        TensorDescriptor[] descriptors = Arrays.stream(names)
                .map(name -> new TensorDescriptor(name, new long[] {1}, TensorDataType.FP32, WeightFormat.FP32, 0, 0))
                .toArray(TensorDescriptor[]::new);
        return new QwenArtifact(null, config, descriptors);
    }

    private static QwenConfig config(
            int layerCount,
            QwenLayerType firstType,
            int experts,
            int expertsPerToken,
            QwenLayerType... remainingTypes) {
        QwenLayerType[] types = new QwenLayerType[layerCount];
        if (layerCount > 0) {
            types[0] = firstType;
            System.arraycopy(remainingTypes, 0, types, 1, Math.min(remainingTypes.length, layerCount - 1));
        }
        return config(layerCount, types, experts, expertsPerToken, experts > 0 ? 4 : 0, experts > 0 ? 2 : 0, 0);
    }

    private static QwenConfig config(int layerCount, QwenLayerType[] types, int experts, int expertsPerToken) {
        return config(layerCount, types, experts, expertsPerToken, experts > 0 ? 4 : 0, experts > 0 ? 2 : 0, 0);
    }

    private static QwenConfig config(
            int layerCount, QwenLayerType firstType, int experts, int expertsPerToken, int mtpLayerCount) {
        QwenLayerType[] types = layerCount == 0 ? new QwenLayerType[0] : new QwenLayerType[] {firstType};
        return config(
                layerCount, types, experts, expertsPerToken, experts > 0 ? 4 : 0, experts > 0 ? 2 : 0, mtpLayerCount);
    }

    private static QwenConfig config(
            int layerCount,
            QwenLayerType[] types,
            int experts,
            int expertsPerToken,
            int moeIntermediateSize,
            int sharedExpertIntermediateSize,
            int mtpLayerCount) {
        return new QwenConfig(
                32_000,
                4_096,
                layerCount,
                32,
                8,
                128,
                11_008,
                16,
                8,
                64,
                128,
                4,
                1.0e-6,
                1_000_000.0,
                1.0,
                32_768,
                "silu",
                types,
                experts,
                expertsPerToken,
                moeIntermediateSize,
                sharedExpertIntermediateSize,
                true,
                false,
                mtpLayerCount);
    }

    private static String[] denseLayerNames(String prefix, boolean moe) {
        return concat(new String[] {prefix + ".input_layernorm.weight"}, attentionNames(prefix), new String[] {
            prefix + ".post_attention_layernorm.weight",
            prefix + ".mlp.gate_proj.weight",
            prefix + ".mlp.up_proj.weight",
            prefix + ".mlp.down_proj.weight"
        });
    }

    private static String[] attentionNames(String prefix) {
        return new String[] {
            prefix + ".self_attn.q_proj.weight",
            prefix + ".self_attn.k_proj.weight",
            prefix + ".self_attn.v_proj.weight",
            prefix + ".self_attn.o_proj.weight",
            prefix + ".self_attn.q_norm.weight",
            prefix + ".self_attn.k_norm.weight"
        };
    }

    private static String[] gdnLayerNames(String prefix) {
        return concat(new String[] {prefix + ".input_layernorm.weight"}, new String[] {
            prefix + ".linear_attn.in_proj_qkv.weight",
            prefix + ".linear_attn.in_proj_z.weight",
            prefix + ".linear_attn.in_proj_b.weight",
            prefix + ".linear_attn.in_proj_a.weight",
            prefix + ".linear_attn.conv1d.weight",
            prefix + ".linear_attn.norm.weight",
            prefix + ".linear_attn.dt_bias",
            prefix + ".linear_attn.A_log",
            prefix + ".linear_attn.out_proj.weight",
            prefix + ".post_attention_layernorm.weight",
            prefix + ".mlp.gate_proj.weight",
            prefix + ".mlp.up_proj.weight",
            prefix + ".mlp.down_proj.weight"
        });
    }

    private static String[] concat(String[]... groups) {
        List<String> names = new ArrayList<>();
        for (String[] group : groups) {
            names.addAll(Arrays.asList(group));
        }
        return names.toArray(String[]::new);
    }

    private static final class FakeGpuMemory implements GpuMemory {

        private final List<Long> allocatedAddresses = new ArrayList<>();
        private final List<Long> freedAddresses = new ArrayList<>();
        private int allocateCalls;
        private int failAtAllocation;
        private long nextAddress = 0x1000;

        @Override
        public long allocate(long byteSize) {
            allocateCalls++;
            long address = nextAddress++;
            if (allocateCalls == failAtAllocation) {
                throw new RuntimeException("synthetic allocation failure");
            }
            allocatedAddresses.add(address);
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void free(long address) {
            freedAddresses.add(address);
        }

        private long addressFor(String name) {
            int index = List.of("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
                    .indexOf(name);
            return 0x1000L + index;
        }
    }
}
