package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.gpu.QwenExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenWeights;
import io.euhedral_execution.inference.core.model_loader.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

final class QwenExecutionFixtures {
    static final int WIDTH = 64;
    static final int VOCABULARY = 8;
    static final long MODEL_ADDRESS = 77;

    static QwenWeights weights() {
        QwenConfig config = new QwenConfig(
                VOCABULARY,
                WIDTH,
                0,
                1,
                1,
                WIDTH,
                WIDTH,
                1,
                1,
                1,
                1,
                1,
                1.0e-6,
                1_000_000.0,
                1.0,
                128,
                "silu",
                new QwenLayerType[0],
                0,
                0,
                0,
                0,
                false,
                false,
                0);
        TensorHandle embedding = q3("embedding", VOCABULARY, MODEL_ADDRESS);
        return new QwenWeights(config, embedding, new QwenLayerWeights[0], norm(), embedding, null);
    }

    static TensorHandle norm() {
        return new TensorHandle(
                "norm",
                new long[] {WIDTH},
                TensorDataType.BF16,
                WeightFormat.BF16,
                WeightLayout.CONTIGUOUS_LE_V1,
                MODEL_ADDRESS + 1,
                WIDTH * Short.BYTES);
    }

    static TensorHandle q3(String name, int outputs, long address) {
        long[] shape = {outputs, WIDTH};
        long bytes = CompactTensorLayout.expectedByteSize(
                shape, TensorDataType.BF16, WeightFormat.Q3_G64_FP16, WeightLayout.ROW_SPLIT_K128_V1);
        return new TensorHandle(
                name,
                shape,
                TensorDataType.BF16,
                WeightFormat.Q3_G64_FP16,
                WeightLayout.ROW_SPLIT_K128_V1,
                address,
                bytes);
    }

    static final class RecordingGpu implements QwenExecutionGpu {
        final AtomicLong nextAddress = new AtomicLong(1000);
        final List<Long> allocations = new ArrayList<>();
        final List<Long> frees = new ArrayList<>();
        final List<String> operations = new ArrayList<>();
        Runnable afterEmbedding = () -> {};
        RuntimeException linearFailure;
        int synchronizations;
        int freeFailures;
        int failAllocationAt;
        int allocationAttempts;
        long failAddressOnce;

        @Override
        public long allocate(long byteSize) {
            if (++allocationAttempts == failAllocationAt) {
                throw new IllegalStateException("injected workspace allocation failure");
            }
            long address = nextAddress.getAndIncrement();
            allocations.add(address);
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {}

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {}

        @Override
        public void free(long address) {
            if (address == failAddressOnce) {
                failAddressOnce = 0;
                throw new IllegalStateException("injected workspace free failure");
            }
            if (freeFailures-- > 0) throw new IllegalStateException("injected free failure");
            frees.add(address);
        }

        @Override
        public void embedQ3(
                long tokenIdsAddress,
                long embeddingAddress,
                long embeddingByteSize,
                long hiddenStateAddress,
                int tokenCount,
                int vocabularySize,
                int hiddenSize) {
            operations.add("embed");
            afterEmbedding.run();
        }

        @Override
        public void rmsNormBf16(
                long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
            operations.add("norm");
        }

        @Override
        public void linearQ3Bf16(
                long inputAddress,
                long weightsAddress,
                long outputAddress,
                int rows,
                int inFeatures,
                int outFeatures,
                long weightsByteSize) {
            operations.add("linear:" + weightsAddress);
            if (linearFailure != null) throw linearFailure;
        }

        @Override
        public void synchronize() {
            synchronizations++;
        }
    }

    private QwenExecutionFixtures() {}
}
