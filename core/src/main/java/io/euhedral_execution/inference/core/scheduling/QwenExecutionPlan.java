package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Immutable instructions and dependency edges for the implemented embedding -> norm -> projection slice.
/// Model weights are borrowed, never owned by an inference quantum.
public final class QwenExecutionPlan {

    public enum Kind {
        EMBEDDING,
        RMS_NORM,
        Q3_LINEAR
    }

    public record Instruction(int id, Kind kind, List<Integer> dependencies, TensorHandle weight, int outputWidth) {
        public Instruction {
            Objects.requireNonNull(kind, "kind");
            dependencies = List.copyOf(dependencies);
            weight = copyHandle(Objects.requireNonNull(weight, "weight"));
            if (id < 0 || outputWidth <= 0) {
                throw new IllegalArgumentException("invalid instruction dimensions");
            }
        }

        @Override
        public TensorHandle weight() {
            return copyHandle(this.weight);
        }
    }

    private final QwenWeights weights;
    private final List<Instruction> instructions;
    private final List<List<Integer>> successors;
    private final List<Integer> projectionWidths;

    public QwenExecutionPlan(QwenWeights weights) {
        this(weights, null, List.of());
    }

    /// A standalone operator slice. Callers must supply semantically valid norm and projection weights;
    /// the compact model's first-layer projections are not a complete inference graph.
    public QwenExecutionPlan(QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        this.weights = Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(weights.config(), "config");
        Objects.requireNonNull(projections, "projections");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0
                || vocabularySize <= 0
                || (normWeight == null && !projections.isEmpty())
                || (normWeight != null && projections.isEmpty())) {
            throw new IllegalArgumentException("invalid operator slice");
        }
        TensorHandle embedding = validate(weights.tokenEmbedding(), vocabularySize, hiddenSize);
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(new Instruction(0, Kind.EMBEDDING, List.of(), embedding, hiddenSize));
        if (normWeight != null) {
            TensorHandle norm = validateNorm(normWeight, hiddenSize);
            nodes.add(new Instruction(1, Kind.RMS_NORM, List.of(0), norm, hiddenSize));
            for (TensorHandle projection : projections) {
                Objects.requireNonNull(projection, "projection");
                long[] shape = projection.shape();
                if (shape == null || shape.length != 2 || shape[0] <= 0 || shape[0] > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("invalid Q3 projection shape");
                }
                nodes.add(new Instruction(
                        nodes.size(),
                        Kind.Q3_LINEAR,
                        List.of(1),
                        validate(projection, shape[0], hiddenSize),
                        Math.toIntExact(shape[0])));
            }
        }
        this.instructions = List.copyOf(nodes);
        this.projectionWidths = nodes.stream()
                .filter(instruction -> instruction.kind() == Kind.Q3_LINEAR)
                .map(Instruction::outputWidth)
                .toList();
        List<List<Integer>> edges = new ArrayList<>();
        for (Instruction ignored : nodes) {
            edges.add(new ArrayList<>());
        }
        for (Instruction instruction : nodes) {
            for (int dependency : instruction.dependencies()) {
                edges.get(dependency).add(instruction.id());
            }
        }
        this.successors = edges.stream().map(List::copyOf).toList();
    }

    public List<Instruction> instructions() {
        return this.instructions;
    }

    public List<Integer> successors(int instructionId) {
        return this.successors.get(instructionId);
    }

    List<Integer> projectionWidths() {
        return this.projectionWidths;
    }

    public QwenWeights weights() {
        return this.weights;
    }

    private static TensorHandle validateNorm(TensorHandle norm, int hiddenSize) {
        long[] shape = norm.shape();
        if (shape == null
                || shape.length != 1
                || shape[0] != hiddenSize
                || norm.deviceAddress() == 0
                || norm.byteSize() != (long) hiddenSize * Short.BYTES
                || norm.dataType() != TensorDataType.BF16
                || norm.format() != WeightFormat.BF16
                || norm.layout() != WeightLayout.CONTIGUOUS_LE_V1) {
            throw new IllegalArgumentException("unsupported input RMSNorm weight");
        }
        return copyHandle(norm);
    }

    private static TensorHandle validate(TensorHandle handle, long rows, int width) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 64 != 0
                || handle.deviceAddress() == 0
                || handle.dataType() != TensorDataType.BF16
                || handle.format() != WeightFormat.Q3_G64_FP16
                || handle.layout() != WeightLayout.ROW_SPLIT_K128_V1
                || handle.byteSize()
                        != CompactTensorLayout.expectedByteSize(
                                shape, handle.dataType(), handle.format(), handle.layout())) {
            throw new IllegalArgumentException("unsupported Q3 weight layout or dimensions");
        }
        return copyHandle(handle);
    }

    private static TensorHandle copyHandle(TensorHandle handle) {
        return new TensorHandle(
                handle.name(),
                handle.shape().clone(),
                handle.dataType(),
                handle.format(),
                handle.layout(),
                handle.deviceAddress(),
                handle.byteSize());
    }
}
