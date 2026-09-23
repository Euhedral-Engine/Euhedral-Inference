package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/// Immutable operation instructions and dependency edges for the loaded Qwen first text layer.
/// Model weights are borrowed; sequence state and quantum workspace have separate owners.
public final class QwenExecutionPlan {

    public enum Kind {
        EMBEDDING,
        RMS_NORM,
        RMS_NORM_UNIT_OFFSET,
        Q3_LINEAR,
        Q4_LINEAR,
        Q5_LINEAR,
        BF16_LINEAR,
        GDN_CONTROL,
        GDN_CONVOLUTION,
        GDN_RECURRENCE,
        GDN_GATED_RMS_NORM,
        RESIDUAL_ADD,
        SWIGLU
    }

    public enum Buffer {
        HIDDEN_STATE,
        MIXER_HIDDEN,
        FINAL_HIDDEN_STATE,
        INPUT_NORMALIZED,
        QK_PROJECTED,
        VALUE_Z_PROJECTED,
        A_PROJECTED,
        B_PROJECTED,
        GDN_G,
        GDN_BETA,
        GDN_CONVOLVED,
        GDN_RECURRENT,
        GDN_NORMALIZED,
        MIXER_DELTA,
        POST_MIXER_NORMALIZED,
        GATE_UP,
        SWIGLU,
        FFN_DELTA,
        SLICE_PROJECTION
    }

    public enum ElementType {
        BF16,
        FP32
    }

    public record BufferSpec(Buffer buffer, int width, ElementType elementType) {
        public BufferSpec {
            Objects.requireNonNull(buffer, "buffer");
            Objects.requireNonNull(elementType, "elementType");
            if (width <= 0) {
                throw new IllegalArgumentException("buffer width must be positive");
            }
        }
    }

    public record Instruction(
            int id,
            Kind kind,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputBuffers,
            List<Buffer> outputBuffers,
            int inputWidth,
            int outputWidth,
            int outputBufferIndex) {
        public Instruction {
            Objects.requireNonNull(kind, "kind");
            dependencies = List.copyOf(dependencies);
            weights = weights.stream().map(QwenExecutionPlan::copyHandle).toList();
            inputBuffers = List.copyOf(inputBuffers);
            outputBuffers = List.copyOf(outputBuffers);
            if (id < 0 || inputWidth < 0 || outputWidth <= 0 || outputBufferIndex < -1) {
                throw new IllegalArgumentException("invalid instruction dimensions");
            }
        }

        public Instruction(int id, Kind kind, List<Integer> dependencies, TensorHandle weight, int outputWidth) {
            this(
                    id,
                    kind,
                    dependencies,
                    weight == null ? List.of() : List.of(weight),
                    List.of(),
                    List.of(),
                    0,
                    outputWidth,
                    -1);
        }

        @Override
        public List<TensorHandle> weights() {
            return this.weights.stream().map(QwenExecutionPlan::copyHandle).toList();
        }

        /// Returns a defensive descriptor copy for inspection outside the instruction executor.
        public TensorHandle weight() {
            return this.weights.isEmpty() ? null : copyHandle(this.weights.getFirst());
        }

        /// Returns a borrowed device address without allocating a defensive `TensorHandle` copy.
        public long weightAddress() {
            return weightAddress(0);
        }

        public long weightAddress(int weightIndex) {
            return this.weights.get(weightIndex).deviceAddress();
        }

        /// Returns payload size without allocating a defensive `TensorHandle` copy.
        public long weightByteSize() {
            return weightByteSize(0);
        }

        public long weightByteSize(int weightIndex) {
            return this.weights.get(weightIndex).byteSize();
        }

        public String weightName(int weightIndex) {
            return this.weights.get(weightIndex).name();
        }
    }

    private record PlanData(
            List<Instruction> instructions,
            List<Integer> projectionWidths,
            List<BufferSpec> bufferSpecs,
            boolean firstLayer) {}

    private final QwenWeights weights;
    private final List<Instruction> instructions;
    private final List<List<Integer>> successors;
    private final List<Integer> projectionWidths;
    private final List<BufferSpec> bufferSpecs;
    private final boolean firstLayer;

    /// Uses the actual loaded layer zero when present; an embedding-only weight view stays embedding-only.
    public QwenExecutionPlan(QwenWeights weights) {
        this(Objects.requireNonNull(weights, "weights"), planFromLoadedWeights(weights));
    }

    /// Builds an embedding-only plan for callers that intentionally validate only token lookup.
    public static QwenExecutionPlan embeddingOnly(QwenWeights weights) {
        Objects.requireNonNull(weights, "weights");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        TensorHandle embedding =
                validateQuantized(weights.tokenEmbedding(), vocabularySize, hiddenSize, WeightFormat.Q3_G64_FP16);
        return new QwenExecutionPlan(weights, embeddingOnly(embedding, hiddenSize));
    }

    /// A standalone operator slice retained for low-level operation validation.
    public QwenExecutionPlan(QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
        this(Objects.requireNonNull(weights, "weights"), operatorSlice(weights, normWeight, projections));
    }

    private QwenExecutionPlan(QwenWeights weights, PlanData data) {
        this.weights = weights;
        this.instructions = List.copyOf(data.instructions());
        this.projectionWidths = List.copyOf(data.projectionWidths());
        this.bufferSpecs = List.copyOf(data.bufferSpecs());
        this.firstLayer = data.firstLayer();
        List<List<Integer>> edges = new ArrayList<>();
        for (int index = 0; index < this.instructions.size(); index++) {
            edges.add(new ArrayList<>());
        }
        for (Instruction instruction : this.instructions) {
            for (int dependency : instruction.dependencies()) {
                if (dependency < 0 || dependency >= instruction.id()) {
                    throw new IllegalArgumentException("instruction dependencies must point to earlier work");
                }
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

    public List<BufferSpec> bufferSpecs() {
        return this.bufferSpecs;
    }

    public boolean hasFirstLayer() {
        return this.firstLayer;
    }

    public int bufferWidth(Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .mapToInt(BufferSpec::width)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public ElementType bufferElementType(Buffer buffer) {
        return this.bufferSpecs.stream()
                .filter(spec -> spec.buffer() == buffer)
                .map(BufferSpec::elementType)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("buffer is not in this plan: " + buffer));
    }

    public QwenWeights weights() {
        return this.weights;
    }

    private static PlanData planFromLoadedWeights(QwenWeights weights) {
        Objects.requireNonNull(weights.config(), "config");
        int hiddenSize = weights.config().hiddenSize();
        int vocabularySize = weights.config().vocabSize();
        if (hiddenSize <= 0 || vocabularySize <= 0) {
            throw new IllegalArgumentException("invalid model dimensions");
        }
        TensorHandle embedding =
                validateQuantized(weights.tokenEmbedding(), vocabularySize, hiddenSize, WeightFormat.Q3_G64_FP16);
        QwenLayerWeights[] layers = weights.layers();
        if (layers == null || layers.length == 0) {
            return embeddingOnly(embedding, hiddenSize);
        }
        return firstLayer(weights, embedding);
    }

    private static PlanData embeddingOnly(TensorHandle embedding, int hiddenSize) {
        return new PlanData(
                List.of(node(
                        0,
                        Kind.EMBEDDING,
                        List.of(),
                        List.of(embedding),
                        List.of(),
                        List.of(Buffer.HIDDEN_STATE),
                        0,
                        hiddenSize)),
                List.of(),
                List.of(),
                false);
    }

    private static PlanData firstLayer(QwenWeights weights, TensorHandle embedding) {
        QwenConfig config = weights.config();
        QwenLayerWeights[] layers = weights.layers();
        if (layers[0] == null || layers[0].index() != 0) {
            throw new IllegalArgumentException("loaded weights do not contain actual text layer zero");
        }
        QwenLayerType[] layerTypes = config.layerTypes();
        if (layerTypes == null || layerTypes.length == 0 || layerTypes[0] != QwenLayerType.GATED_DELTA_NET) {
            throw new IllegalArgumentException("layer zero is not the loaded GDN topology");
        }
        if (!(layers[0].mixer() instanceof QwenCompactGatedDeltaNetWeights gdn)
                || !(layers[0].ffn() instanceof QwenCompactDenseFfnWeights ffn)) {
            throw new IllegalArgumentException("layer zero must use compact GDN and dense FFN weights");
        }
        int hidden = config.hiddenSize();
        int intermediate = config.intermediateSize();
        int keyHeads = config.linearNumKeyHeads();
        int valueHeads = config.linearNumValueHeads();
        int keyWidth = config.linearKeyHeadDim();
        int valueWidthPerHead = config.linearValueHeadDim();
        int kernelWidth = config.linearConvKernelDim();
        if (hidden <= 0
                || intermediate <= 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyWidth != 128
                || valueWidthPerHead != 128
                || kernelWidth < 2
                || !Double.isFinite(config.rmsNormEpsilon())
                || config.rmsNormEpsilon() <= 0) {
            throw new IllegalArgumentException("unsupported layer-zero GDN geometry");
        }
        int keyProjected = Math.multiplyExact(Math.multiplyExact(2, keyHeads), keyWidth);
        int valueProjected = Math.multiplyExact(valueHeads, valueWidthPerHead);
        int valueZProjected = Math.multiplyExact(2, valueProjected);
        int convolutionChannels = Math.addExact(keyProjected, valueProjected);

        TensorHandle inputNorm = validateNorm(layers[0].inputNorm(), hidden);
        TensorHandle postNorm = validateNorm(layers[0].postAttentionNorm(), hidden);
        TensorHandle aLog = validateFp32Vector(gdn.aLog(), valueHeads);
        TensorHandle dtBias = validateFp32Vector(gdn.dtBias(), valueHeads);
        TensorHandle convolution = validateBf16Matrix(gdn.convolution(), kernelWidth, convolutionChannels);
        TensorHandle aProjection = validateBf16Matrix(gdn.aProjection(), valueHeads, hidden);
        TensorHandle bProjection = validateBf16Matrix(gdn.bProjection(), valueHeads, hidden);
        TensorHandle queryKey = validateQuantized(gdn.queryKey(), keyProjected, hidden, WeightFormat.Q4_G64_FP16);
        TensorHandle valueZ = validateQuantized(gdn.valueZ(), valueZProjected, hidden, WeightFormat.Q5_G64_FP16);
        TensorHandle gdnNorm = validateNorm(gdn.norm(), valueWidthPerHead);
        TensorHandle gdnOutput = validateQuantized(gdn.output(), hidden, valueProjected, WeightFormat.Q3_G64_FP16);
        TensorHandle gateUp =
                validateQuantized(ffn.gateUp(), Math.multiplyExact(2, intermediate), hidden, WeightFormat.Q3_G64_FP16);
        TensorHandle down = validateQuantized(ffn.down(), hidden, intermediate, WeightFormat.Q3_G64_FP16);

        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0, Kind.EMBEDDING, List.of(), List.of(embedding), List.of(), List.of(Buffer.HIDDEN_STATE), 0, hidden));
        nodes.add(node(
                1,
                Kind.RMS_NORM_UNIT_OFFSET,
                List.of(0),
                List.of(inputNorm),
                List.of(Buffer.HIDDEN_STATE),
                List.of(Buffer.INPUT_NORMALIZED),
                hidden,
                hidden));
        nodes.add(node(
                2,
                Kind.Q4_LINEAR,
                List.of(1),
                List.of(queryKey),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.QK_PROJECTED),
                hidden,
                keyProjected));
        nodes.add(node(
                3,
                Kind.Q5_LINEAR,
                List.of(1),
                List.of(valueZ),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.VALUE_Z_PROJECTED),
                hidden,
                valueZProjected));
        nodes.add(node(
                4,
                Kind.BF16_LINEAR,
                List.of(1),
                List.of(aProjection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.A_PROJECTED),
                hidden,
                valueHeads));
        nodes.add(node(
                5,
                Kind.BF16_LINEAR,
                List.of(1),
                List.of(bProjection),
                List.of(Buffer.INPUT_NORMALIZED),
                List.of(Buffer.B_PROJECTED),
                hidden,
                valueHeads));
        nodes.add(node(
                6,
                Kind.GDN_CONTROL,
                List.of(4, 5),
                List.of(aLog, dtBias),
                List.of(Buffer.A_PROJECTED, Buffer.B_PROJECTED),
                List.of(Buffer.GDN_G, Buffer.GDN_BETA),
                valueHeads,
                valueHeads));
        nodes.add(node(
                7,
                Kind.GDN_CONVOLUTION,
                List.of(2, 3),
                List.of(convolution),
                List.of(Buffer.QK_PROJECTED, Buffer.VALUE_Z_PROJECTED),
                List.of(Buffer.GDN_CONVOLVED),
                convolutionChannels,
                convolutionChannels));
        nodes.add(node(
                8,
                Kind.GDN_RECURRENCE,
                List.of(6, 7),
                List.of(),
                List.of(Buffer.GDN_CONVOLVED, Buffer.GDN_G, Buffer.GDN_BETA),
                List.of(Buffer.GDN_RECURRENT),
                convolutionChannels,
                valueProjected));
        nodes.add(node(
                9,
                Kind.GDN_GATED_RMS_NORM,
                List.of(8, 3),
                List.of(gdnNorm),
                List.of(Buffer.GDN_RECURRENT, Buffer.VALUE_Z_PROJECTED),
                List.of(Buffer.GDN_NORMALIZED),
                valueProjected,
                valueProjected));
        nodes.add(node(
                10,
                Kind.Q3_LINEAR,
                List.of(9),
                List.of(gdnOutput),
                List.of(Buffer.GDN_NORMALIZED),
                List.of(Buffer.MIXER_DELTA),
                valueProjected,
                hidden));
        nodes.add(node(
                11,
                Kind.RESIDUAL_ADD,
                List.of(0, 10),
                List.of(),
                List.of(Buffer.HIDDEN_STATE, Buffer.MIXER_DELTA),
                List.of(Buffer.MIXER_HIDDEN),
                hidden,
                hidden));
        nodes.add(node(
                12,
                Kind.RMS_NORM_UNIT_OFFSET,
                List.of(11),
                List.of(postNorm),
                List.of(Buffer.MIXER_HIDDEN),
                List.of(Buffer.POST_MIXER_NORMALIZED),
                hidden,
                hidden));
        nodes.add(node(
                13,
                Kind.Q3_LINEAR,
                List.of(12),
                List.of(gateUp),
                List.of(Buffer.POST_MIXER_NORMALIZED),
                List.of(Buffer.GATE_UP),
                hidden,
                2 * intermediate));
        nodes.add(node(
                14,
                Kind.SWIGLU,
                List.of(13),
                List.of(),
                List.of(Buffer.GATE_UP),
                List.of(Buffer.SWIGLU),
                2 * intermediate,
                intermediate));
        nodes.add(node(
                15,
                Kind.Q3_LINEAR,
                List.of(14),
                List.of(down),
                List.of(Buffer.SWIGLU),
                List.of(Buffer.FFN_DELTA),
                intermediate,
                hidden));
        nodes.add(node(
                16,
                Kind.RESIDUAL_ADD,
                List.of(11, 15),
                List.of(),
                List.of(Buffer.MIXER_HIDDEN, Buffer.FFN_DELTA),
                List.of(Buffer.FINAL_HIDDEN_STATE),
                hidden,
                hidden));

        List<BufferSpec> buffers = List.of(
                spec(Buffer.HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.MIXER_HIDDEN, hidden, ElementType.BF16),
                spec(Buffer.FINAL_HIDDEN_STATE, hidden, ElementType.BF16),
                spec(Buffer.INPUT_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.QK_PROJECTED, keyProjected, ElementType.BF16),
                spec(Buffer.VALUE_Z_PROJECTED, valueZProjected, ElementType.BF16),
                spec(Buffer.A_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.B_PROJECTED, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_G, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_BETA, valueHeads, ElementType.FP32),
                spec(Buffer.GDN_CONVOLVED, convolutionChannels, ElementType.BF16),
                spec(Buffer.GDN_RECURRENT, valueProjected, ElementType.BF16),
                spec(Buffer.GDN_NORMALIZED, valueProjected, ElementType.BF16),
                spec(Buffer.MIXER_DELTA, hidden, ElementType.BF16),
                spec(Buffer.POST_MIXER_NORMALIZED, hidden, ElementType.BF16),
                spec(Buffer.GATE_UP, 2 * intermediate, ElementType.BF16),
                spec(Buffer.SWIGLU, intermediate, ElementType.BF16),
                spec(Buffer.FFN_DELTA, hidden, ElementType.BF16));
        return new PlanData(List.copyOf(nodes), List.of(), buffers, true);
    }

    private static PlanData operatorSlice(
            QwenWeights weights, TensorHandle normWeight, List<TensorHandle> projections) {
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
        TensorHandle embedding =
                validateQuantized(weights.tokenEmbedding(), vocabularySize, hiddenSize, WeightFormat.Q3_G64_FP16);
        List<Instruction> nodes = new ArrayList<>();
        nodes.add(node(
                0,
                Kind.EMBEDDING,
                List.of(),
                List.of(embedding),
                List.of(),
                List.of(Buffer.HIDDEN_STATE),
                0,
                hiddenSize));
        if (normWeight != null) {
            TensorHandle norm = validateNorm(normWeight, hiddenSize);
            nodes.add(node(
                    1,
                    Kind.RMS_NORM,
                    List.of(0),
                    List.of(norm),
                    List.of(Buffer.HIDDEN_STATE),
                    List.of(Buffer.INPUT_NORMALIZED),
                    hiddenSize,
                    hiddenSize));
            for (int index = 0; index < projections.size(); index++) {
                TensorHandle projection = Objects.requireNonNull(projections.get(index), "projection");
                long[] shape = projection.shape();
                if (shape == null || shape.length != 2 || shape[0] <= 0 || shape[0] > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("invalid Q3 projection shape");
                }
                nodes.add(new Instruction(
                        nodes.size(),
                        Kind.Q3_LINEAR,
                        List.of(1),
                        List.of(validateQuantized(projection, shape[0], hiddenSize, WeightFormat.Q3_G64_FP16)),
                        List.of(Buffer.INPUT_NORMALIZED),
                        List.of(Buffer.SLICE_PROJECTION),
                        hiddenSize,
                        Math.toIntExact(shape[0]),
                        index));
            }
        }
        List<Integer> widths = nodes.stream()
                .filter(instruction -> instruction.kind() == Kind.Q3_LINEAR)
                .map(Instruction::outputWidth)
                .toList();
        return new PlanData(List.copyOf(nodes), widths, List.of(), false);
    }

    private static Instruction node(
            int id,
            Kind kind,
            List<Integer> dependencies,
            List<TensorHandle> weights,
            List<Buffer> inputs,
            List<Buffer> outputs,
            int inputWidth,
            int outputWidth) {
        return new Instruction(id, kind, dependencies, weights, inputs, outputs, inputWidth, outputWidth, -1);
    }

    private static BufferSpec spec(Buffer buffer, int width, ElementType type) {
        return new BufferSpec(buffer, width, type);
    }

    private static TensorHandle validateNorm(TensorHandle norm, int width) {
        return validateDirect(norm, new long[] {width}, WeightFormat.BF16, (long) width * Short.BYTES);
    }

    private static TensorHandle validateFp32Vector(TensorHandle vector, int width) {
        return validateDirect(vector, new long[] {width}, WeightFormat.FP32, (long) width * Float.BYTES);
    }

    private static TensorHandle validateBf16Matrix(TensorHandle matrix, int rows, int columns) {
        return validateDirect(
                matrix, new long[] {rows, columns}, WeightFormat.BF16, (long) rows * columns * Short.BYTES);
    }

    private static TensorHandle validateDirect(
            TensorHandle handle, long[] expectedShape, WeightFormat format, long expectedByteSize) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || !java.util.Arrays.equals(shape, expectedShape)
                || handle.deviceAddress() == 0
                || handle.dataType() != TensorDataType.BF16
                || handle.format() != format
                || handle.layout() != WeightLayout.CONTIGUOUS_LE_V1
                || handle.byteSize() != expectedByteSize) {
            throw new IllegalArgumentException("unsupported direct weight layout or dimensions: " + handle.name());
        }
        return copyHandle(handle);
    }

    private static TensorHandle validateQuantized(TensorHandle handle, long rows, int width, WeightFormat format) {
        Objects.requireNonNull(handle, "weight");
        long[] shape = handle.shape();
        if (shape == null
                || shape.length != 2
                || shape[0] != rows
                || shape[1] != width
                || width % 64 != 0
                || handle.deviceAddress() == 0
                || handle.dataType() != TensorDataType.BF16
                || handle.format() != format
                || handle.layout() != WeightLayout.ROW_SPLIT_K128_V1
                || handle.byteSize()
                        != CompactTensorLayout.expectedByteSize(
                                shape, handle.dataType(), handle.format(), handle.layout())) {
            throw new IllegalArgumentException("unsupported quantized weight layout or dimensions: " + handle.name());
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
