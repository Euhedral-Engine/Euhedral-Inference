package io.euhedral_execution.inference.core.gpu;

import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.assertBf16Equals;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.bf16ToFloat;
import static io.euhedral_execution.inference.core.gpu.CudaGpuOperationsIntegrationTest.download;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.model_loader.QwenWeightLoader;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.scheduling.AttentionSequenceStates;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionRunner;
import io.euhedral_execution.inference.core.scheduling.GdnSequenceStates;
import io.euhedral_execution.inference.core.scheduling.QwenSequenceState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class QwenFullModelCudaIntegrationTest {

    private static final int INITIAL_TOKEN = 1814;
    private static final long VRAM_RESTORE_TOLERANCE = 128L * 1024L * 1024L;
    private static final float HIDDEN_TOLERANCE = 1.0f;
    private static final float LOGIT_TOLERANCE = 1.0f;
    private static final List<QwenExecutionPlan.Buffer> FIRST_LAYER_BOUNDARIES = List.of(
            QwenExecutionPlan.Buffer.HIDDEN_STATE,
            QwenExecutionPlan.Buffer.INPUT_NORMALIZED,
            QwenExecutionPlan.Buffer.QK_PROJECTED,
            QwenExecutionPlan.Buffer.VALUE_Z_PROJECTED,
            QwenExecutionPlan.Buffer.GDN_CONVOLVED,
            QwenExecutionPlan.Buffer.GDN_RECURRENT,
            QwenExecutionPlan.Buffer.GDN_NORMALIZED,
            QwenExecutionPlan.Buffer.MIXER_DELTA,
            QwenExecutionPlan.Buffer.MIXER_HIDDEN,
            QwenExecutionPlan.Buffer.POST_MIXER_NORMALIZED,
            QwenExecutionPlan.Buffer.GATE_UP,
            QwenExecutionPlan.Buffer.SWIGLU,
            QwenExecutionPlan.Buffer.FFN_DELTA,
            QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE);

    @Test
    @Timeout(value = 1200, unit = TimeUnit.SECONDS)
    void realCompactQwenRunsAllLayersAndMatchesCpuReferenceAcrossBoundaries() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact"));
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(artifactPath), "compact Qwen artifact is missing: " + artifactPath);
        QwenArtifact artifact = QwenArtifactReader.read(artifactPath);
        assertEquals(64, artifact.config().numHiddenLayers());
        assertTrue(Arrays.asList(artifact.config().layerTypes()).contains(QwenLayerType.FULL_ATTENTION));
        assertTrue(Arrays.asList(artifact.config().layerTypes()).contains(QwenLayerType.GATED_DELTA_NET));

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            long freeBefore = gpu.deviceMemoryInfo().freeBytes();
            QwenWeights weights = QwenWeightLoader.load(artifactPath, artifact, gpu);
            List<Long> modelAddresses = weights.runtimeObjects().values().stream()
                    .map(handle -> handle.deviceAddress())
                    .distinct()
                    .collect(java.util.stream.Collectors.toCollection(ArrayList::new));
            Throwable failure = null;
            QwenSequenceState prefixSequence = new QwenSequenceState(501);
            QwenSequenceState mixedSequence = new QwenSequenceState(502);
            QwenSequenceState stateSequence = new QwenSequenceState(503);
            QwenSequenceState referenceSequence = new QwenSequenceState(504);
            QwenSequenceState isolationSequence = new QwenSequenceState(505);
            List<RunResult> runs = new ArrayList<>();
            try {
                QwenExecutionPlan plan = new QwenExecutionPlan(weights);
                QwenExecutionPlan firstLayerPlan = QwenExecutionPlan.prefix(weights, 1);
                QwenExecutionPlan mixedPlan = QwenExecutionPlan.prefix(weights, 4);
                int fullAttentionLayers = (int) Arrays.stream(weights.config().layerTypes())
                        .filter(type -> type == QwenLayerType.FULL_ATTENTION)
                        .count();
                assertEquals(
                        64,
                        plan.instructions().stream()
                                        .mapToInt(QwenExecutionPlan.Instruction::layerIndex)
                                        .filter(index -> index >= 0)
                                        .max()
                                        .orElseThrow()
                                + 1);
                assertEquals(
                        fullAttentionLayers,
                        plan.instructions().stream()
                                .filter(instruction -> instruction.kind() == QwenExecutionPlan.Kind.ATTENTION_CAUSAL)
                                .count());
                for (int layerIndex = 0; layerIndex < weights.layers().length; layerIndex++) {
                    int selectedLayer = layerIndex;
                    long expectedWeight =
                            weights.layers()[layerIndex].inputNorm().deviceAddress();
                    assertTrue(plan.instructions().stream()
                            .anyMatch(instruction -> instruction.layerIndex() == selectedLayer
                                    && instruction.weightAddress() == expectedWeight));
                }

                QwenFullModelCpuReference.Result reference = QwenFullModelCpuReference.run(weights, gpu, INITIAL_TOKEN);
                QwenFirstLayerCpuReference.Result firstLayerReference =
                        QwenFirstLayerCpuReference.run(weights, gpu, INITIAL_TOKEN);

                RunResult firstLayer = execute(
                        gpu,
                        firstLayerPlan,
                        prefixSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        FIRST_LAYER_BOUNDARIES);
                runs.add(firstLayer);
                assertSuccessful(firstLayer);
                assertBf16Equals(
                        firstLayerReference.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        firstLayer.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        0.08f);
                for (QwenExecutionPlan.Buffer boundary : FIRST_LAYER_BOUNDARIES) {
                    assertBf16Equals(
                            firstLayerReference.buffers().get(boundary),
                            firstLayer.buffers().get(boundary),
                            0.08f);
                }

                RunResult mixed = execute(
                        gpu,
                        mixedPlan,
                        mixedSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                runs.add(mixed);
                assertSuccessful(mixed);
                assertBf16Equals(
                        reference.layerOutputs().get(3),
                        mixed.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        HIDDEN_TOLERANCE);

                RunResult prefill = execute(
                        gpu,
                        plan,
                        stateSequence,
                        QwenExecutionContext.ExecutionKind.PREFILL,
                        0,
                        new int[] {INITIAL_TOKEN, 26},
                        List.of(QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                runs.add(prefill);
                assertSuccessful(prefill);
                assertEquals(2, stateSequence.currentTokenPosition());
                GdnSequenceStates recurrent = (GdnSequenceStates) stateSequence.recurrentState();
                AttentionSequenceStates attention = (AttentionSequenceStates) stateSequence.kvCacheState();
                long firstRecurrentAddress = recurrent.forLayer(0).recurrentStateAddress();
                long firstConvolutionAddress = recurrent.forLayer(0).convolutionStateAddress();
                long recurrentBytes = (long) weights.config().linearNumValueHeads()
                        * weights.config().linearKeyHeadDim()
                        * weights.config().linearValueHeadDim()
                        * Float.BYTES;
                byte[] recurrentAfterPrefill = readDeviceBytes(gpu, firstRecurrentAddress, recurrentBytes);
                int attentionLayers = 0;
                for (int layerIndex = 0; layerIndex < weights.config().layerTypes().length; layerIndex++) {
                    if (weights.config().layerTypes()[layerIndex] != QwenLayerType.FULL_ATTENTION) continue;
                    assertEquals(2, attention.forLayer(layerIndex).length());
                    attentionLayers++;
                }
                assertEquals(fullAttentionLayers, attentionLayers);

                RunResult decode = execute(
                        gpu,
                        plan,
                        stateSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        2,
                        new int[] {13},
                        List.of(QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                runs.add(decode);
                assertSuccessful(decode);
                assertEquals(3, stateSequence.currentTokenPosition());
                assertTrue(
                        !Arrays.equals(
                                recurrentAfterPrefill, readDeviceBytes(gpu, firstRecurrentAddress, recurrentBytes)),
                        "GDN recurrent state did not advance in the decode quantum");
                assertTrue(firstConvolutionAddress != 0);
                for (int layerIndex = 0; layerIndex < weights.config().layerTypes().length; layerIndex++) {
                    if (weights.config().layerTypes()[layerIndex] == QwenLayerType.FULL_ATTENTION) {
                        assertEquals(3, attention.forLayer(layerIndex).length());
                    }
                }

                RunResult cleanSequence = execute(
                        gpu,
                        plan,
                        referenceSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of(
                                QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE,
                                QwenExecutionPlan.Buffer.FINAL_NORMALIZED));
                runs.add(cleanSequence);
                assertSuccessful(cleanSequence);
                assertEquals(1, referenceSequence.currentTokenPosition());
                assertNotSame(stateSequence.recurrentState(), referenceSequence.recurrentState());
                assertNotSame(stateSequence.kvCacheState(), referenceSequence.kvCacheState());
                assertEquals(
                        1,
                        ((AttentionSequenceStates) referenceSequence.kvCacheState())
                                .forLayer(3)
                                .length());
                assertBf16Equals(
                        reference.layerOutputs().get(63),
                        cleanSequence.buffers().get(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE),
                        HIDDEN_TOLERANCE);
                assertBf16Equals(
                        reference.finalNormalized(),
                        cleanSequence.buffers().get(QwenExecutionPlan.Buffer.FINAL_NORMALIZED),
                        HIDDEN_TOLERANCE);
                assertBf16Equals(reference.logits(), cleanSequence.logits(), LOGIT_TOLERANCE);
                for (short value : cleanSequence.logits()) {
                    assertTrue(
                            Float.isFinite(bf16ToFloat(value)),
                            "final vocabulary projection produced a non-finite logit");
                }

                RunResult isolatedSequence = execute(
                        gpu,
                        plan,
                        isolationSequence,
                        QwenExecutionContext.ExecutionKind.DECODE,
                        0,
                        new int[] {INITIAL_TOKEN},
                        List.of(QwenExecutionPlan.Buffer.FINAL_HIDDEN_STATE));
                runs.add(isolatedSequence);
                assertSuccessful(isolatedSequence);
                assertArrayEquals(cleanSequence.logits(), isolatedSequence.logits());
                assertNotEquals(
                        ((GdnSequenceStates) stateSequence.recurrentState())
                                .forLayer(0)
                                .recurrentStateAddress(),
                        ((GdnSequenceStates) referenceSequence.recurrentState())
                                .forLayer(0)
                                .recurrentStateAddress());

                GdnSequenceStates completedRecurrent = recurrent;
                AttentionSequenceStates completedAttention = attention;
                stateSequence.complete();
                prefixSequence.complete();
                mixedSequence.complete();
                referenceSequence.complete();
                isolationSequence.complete();
                assertThrows(IllegalStateException.class, () -> completedRecurrent.forLayer(0));
                assertThrows(IllegalStateException.class, () -> completedAttention.forLayer(3));
                for (RunResult run : runs) assertTrue(run.context().workspace().isClosed());
            } catch (Throwable executionFailure) {
                failure = executionFailure;
            } finally {
                for (QwenSequenceState sequence :
                        List.of(isolationSequence, referenceSequence, stateSequence, mixedSequence, prefixSequence)) {
                    try {
                        sequence.complete();
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
                for (RunResult run : runs) {
                    try {
                        run.closeLogits();
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
                for (int index = modelAddresses.size() - 1; index >= 0; index--) {
                    try {
                        gpu.free(modelAddresses.get(index));
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
            }
            long freeAfter = gpu.deviceMemoryInfo().freeBytes();
            if (Math.abs(freeAfter - freeBefore) > VRAM_RESTORE_TOLERANCE) {
                IllegalStateException cleanupFailure = new IllegalStateException(
                        "full model test leaked device memory: before=" + freeBefore + ", after=" + freeAfter);
                if (failure == null) failure = cleanupFailure;
                else failure.addSuppressed(cleanupFailure);
            }
            if (failure != null) {
                if (failure instanceof Exception exception) throw exception;
                if (failure instanceof Error error) throw error;
                throw new IllegalStateException(failure);
            }
        }
    }

    private static RunResult execute(
            CudaGpuMemory gpu,
            QwenExecutionPlan plan,
            QwenSequenceState sequence,
            QwenExecutionContext.ExecutionKind kind,
            long startPosition,
            int[] tokenIds,
            List<QwenExecutionPlan.Buffer> capturedBuffers)
            throws Exception {
        EnumMap<QwenExecutionPlan.Buffer, short[]> buffers = new EnumMap<>(QwenExecutionPlan.Buffer.class);
        AtomicReference<short[]> logits = new AtomicReference<>();
        AtomicReference<QwenExecutionContext> completedContext = new AtomicReference<>();
        QwenExecutionRunner runner = new QwenExecutionRunner(plan, gpu, context -> {
            completedContext.set(context);
            try (Arena arena = Arena.ofShared()) {
                for (QwenExecutionPlan.Buffer buffer : capturedBuffers) {
                    if (!context.workspace().hasBuffer(buffer)) continue;
                    buffers.put(
                            buffer,
                            download(
                                    gpu,
                                    arena,
                                    context.workspace().address(buffer),
                                    Math.toIntExact(context.workspace().bufferByteSize(buffer) / Short.BYTES)));
                }
                context.logitsOutput()
                        .ifPresent(deviceLogits -> logits.set(download(
                                gpu,
                                arena,
                                deviceLogits.deviceAddress(),
                                Math.multiplyExact(deviceLogits.tokenCount(), deviceLogits.vocabularySize()))));
            }
        });
        new DefaultExecutor().input(runner);
        try {
            var outcome = runner.submit(new QwenExecutionContext(plan, sequence, kind, startPosition, tokenIds));
            runner.request(plan.instructions().size());
            QwenExecutionContext.Outcome result = outcome.get(600, TimeUnit.SECONDS);
            RunResult run = new RunResult(completedContext.get(), buffers, logits.get());
            if (result.status() != QwenExecutionContext.Status.SUCCESS) {
                throw new AssertionError("Euhedral graph failed: " + result.failure());
            }
            return run;
        } finally {
            runner.completeGracefully();
        }
    }

    private static void assertSuccessful(RunResult run) {
        assertNotNull(run.context());
        assertTrue(run.context().workspace().isClosed());
    }

    private static byte[] readDeviceBytes(CudaGpuMemory gpu, long address, long byteSize) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(byteSize, Integer.BYTES);
            gpu.copyDeviceToHost(host, address, byteSize);
            return host.toArray(ValueLayout.JAVA_BYTE);
        }
    }

    private record RunResult(
            QwenExecutionContext context, EnumMap<QwenExecutionPlan.Buffer, short[]> buffers, short[] logits) {
        private void closeLogits() {
            context.logitsOutput().ifPresent(deviceLogits -> deviceLogits.close());
        }
    }
}
