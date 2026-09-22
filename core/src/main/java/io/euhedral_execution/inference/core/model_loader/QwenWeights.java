package io.euhedral_execution.inference.core.model_loader;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenLayerWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenMtpWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;

public record QwenWeights(
        QwenConfig config,
        TensorHandle tokenEmbedding,
        QwenLayerWeights[] layers,
        TensorHandle finalNorm,
        TensorHandle lmHead,
        QwenMtpWeights mtp) {}
