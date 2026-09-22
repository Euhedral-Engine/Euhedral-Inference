package io.euhedral_execution.inference.core.model_loader.layer_weights;

public record QwenGatedDeltaNetWeights(
        TensorHandle inProjQkv,
        TensorHandle inProjZ,
        TensorHandle inProjB,
        TensorHandle inProjA,
        TensorHandle convid,
        TensorHandle norm,
        TensorHandle dtBias,
        TensorHandle aLog,
        TensorHandle outProj)
        implements QwenMixerWeights {}
