package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.PipelineFrame;
import io.euhedral_execution.core.ingest.PipelineRunner;
import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactMtpAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/// Immutable execution definition for one loaded Qwen model and its non-owning GPU runtime.
///
/// The definition contains one reusable Euhedral stage for preparation, each transformer layer,
/// final normalization/LM-head work, and terminal result capture. A transformer layer remains one
/// stage even when its future implementation contains several CUDA operations.
public final class QwenExecutionPlan {

    public record LayerOperation(int index, QwenLayerType type) {

        public LayerOperation {
            if (index < 0) {
                throw new IllegalArgumentException("layer index must be non-negative");
            }
            Objects.requireNonNull(type, "type");
        }
    }

    private final QwenWeights weights;
    private final QwenExecutionGpu gpu;
    private final List<QwenLayerWeights> layerWeights;
    private final List<LayerOperation> layerOperations;
    private final PipelineFrame.Builder<QwenExecutionContext, QwenExecutionContext> pipelineDefinition;

    public QwenExecutionPlan(QwenWeights weights, QwenExecutionGpu gpu) {
        this.weights = snapshotWeights(Objects.requireNonNull(weights, "weights"));
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.layerOperations = preselectLayerOperations(this.weights);
        this.layerWeights = immutableLayerView(this.weights);
        this.pipelineDefinition = buildPipeline(this.layerOperations);
    }

    QwenWeights weights() {
        return this.weights;
    }

    QwenExecutionGpu gpu() {
        return this.gpu;
    }

    public List<QwenLayerWeights> layerWeights() {
        return this.layerWeights;
    }

    public List<LayerOperation> layerOperations() {
        return this.layerOperations;
    }

    /// Returns the immutable Euhedral frame definition used by every runner created from this plan.
    public PipelineFrame.Builder<QwenExecutionContext, QwenExecutionContext> pipelineDefinition() {
        return this.pipelineDefinition;
    }

    /// Creates a runner whose terminal consumer captures the context result.
    public QwenExecutionRunner newRunner() {
        return newRunner(context -> {});
    }

    /// Creates a runner that captures the context result before invoking the supplied consumer.
    public QwenExecutionRunner newRunner(Consumer<? super QwenExecutionContext> terminalConsumer) {
        Objects.requireNonNull(terminalConsumer, "terminalConsumer");
        PipelineRunner<QwenExecutionContext> pipelineRunner = new PipelineRunner<>(
                this.pipelineDefinition,
                context -> {
                    context.captureTerminal();
                    try {
                        terminalConsumer.accept(context);
                        context.checkTerminalBoundary();
                    } catch (io.euhedral_execution.core.frames.AbstractFrame.CancelSignal cancellation) {
                        throw cancellation;
                    } catch (RuntimeException failure) {
                        context.invalidateAfterTerminalConsumerFailure(failure);
                        throw failure;
                    } catch (Error failure) {
                        context.invalidateAfterTerminalConsumerFailure(failure);
                        throw new TerminalConsumerFailure(failure);
                    }
                },
                true);
        return new QwenExecutionRunner(pipelineRunner);
    }

    private static QwenWeights snapshotWeights(QwenWeights source) {
        QwenLayerWeights[] layers = source.layers();
        return new QwenWeights(
                snapshotConfig(source.config()),
                source.tokenEmbedding(),
                layers == null ? null : layers.clone(),
                source.finalNorm(),
                source.lmHead(),
                source.mtp(),
                source.runtimeObjects());
    }

    private static List<QwenLayerWeights> immutableLayerView(QwenWeights weights) {
        QwenLayerWeights[] layers = weights.layers();
        return layers == null ? List.of() : List.copyOf(Arrays.asList(layers));
    }

    private static QwenConfig snapshotConfig(QwenConfig source) {
        if (source == null) {
            return null;
        }
        QwenLayerType[] layerTypes = source.layerTypes();
        return new QwenConfig(
                source.vocabSize(),
                source.hiddenSize(),
                source.numHiddenLayers(),
                source.numAttentionHeads(),
                source.numKeyValueHeads(),
                source.attentionHeadDim(),
                source.intermediateSize(),
                source.linearNumKeyHeads(),
                source.linearNumValueHeads(),
                source.linearKeyHeadDim(),
                source.linearValueHeadDim(),
                source.linearConvKernelDim(),
                source.rmsNormEpsilon(),
                source.ropeTheta(),
                source.partialRotaryFactor(),
                source.maxPositionEmbeddings(),
                source.hiddenActivation(),
                layerTypes == null ? null : layerTypes.clone(),
                source.numExperts(),
                source.numExpertsPerToken(),
                source.moeIntermediateSize(),
                source.sharedExpertIntermediateSize(),
                source.tieWordEmbeddings(),
                source.attentionOutputGate(),
                source.mtpLayerCount());
    }

    private static List<LayerOperation> preselectLayerOperations(QwenWeights weights) {
        if (weights.config() == null) {
            throw new IllegalArgumentException("Qwen weights have no configuration");
        }
        if (weights.layers() == null) {
            throw new IllegalArgumentException("Qwen weights have no transformer layers");
        }
        QwenLayerType[] configuredTypes = weights.config().layerTypes();
        if (configuredTypes == null || configuredTypes.length != weights.layers().length) {
            throw new IllegalArgumentException("Qwen layer topology does not match model configuration");
        }

        List<LayerOperation> operations = new ArrayList<>(weights.layers().length);
        for (int index = 0; index < weights.layers().length; index++) {
            QwenLayerWeights layer = weights.layers()[index];
            if (layer == null) {
                throw new IllegalArgumentException("Qwen layer " + index + " is missing");
            }
            if (layer.index() != index) {
                throw new IllegalArgumentException(
                        "Qwen layer index " + layer.index() + " is stored at position " + index);
            }
            QwenLayerType actualType = mixerType(layer, index);
            if (configuredTypes[index] != actualType) {
                throw new IllegalArgumentException("Qwen layer " + index + " is " + actualType
                        + " but configuration declares " + configuredTypes[index]);
            }
            operations.add(new LayerOperation(index, actualType));
        }
        return List.copyOf(operations);
    }

    private static QwenLayerType mixerType(QwenLayerWeights layer, int index) {
        if (layer.mixer() instanceof QwenAttentionWeights) {
            return QwenLayerType.FULL_ATTENTION;
        }
        if (layer.mixer() instanceof QwenGatedDeltaNetWeights) {
            return QwenLayerType.GATED_DELTA_NET;
        }
        if (layer.mixer() instanceof QwenCompactAttentionWeights
                || layer.mixer() instanceof QwenCompactMtpAttentionWeights) {
            return QwenLayerType.FULL_ATTENTION;
        }
        if (layer.mixer() instanceof QwenCompactGatedDeltaNetWeights) {
            return QwenLayerType.GATED_DELTA_NET;
        }
        throw new IllegalArgumentException("Qwen layer " + index + " has an unsupported mixer");
    }

    private static final class TerminalConsumerFailure extends RuntimeException {

        private TerminalConsumerFailure(Error cause) {
            super("Terminal consumer failed with an Error", cause);
        }
    }

    private PipelineFrame.Builder<QwenExecutionContext, QwenExecutionContext> buildPipeline(
            List<LayerOperation> operations) {
        PipelineFrame.Builder<QwenExecutionContext, QwenExecutionContext> builder =
                PipelineFrame.<QwenExecutionContext>builder().fanOut(context -> context.prepare(this));
        for (LayerOperation operation : operations) {
            builder = builder.fanOut(context -> context.executeLayer(operation));
        }
        return builder.fanOut(QwenExecutionContext::executeFinal);
    }
}
