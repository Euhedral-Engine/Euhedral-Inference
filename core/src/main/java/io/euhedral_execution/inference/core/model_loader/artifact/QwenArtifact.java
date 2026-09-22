package io.euhedral_execution.inference.core.model_loader.artifact;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;

public record QwenArtifact(QwenArtifactHeader header, QwenConfig config, TensorDescriptor[] tensors) {}
