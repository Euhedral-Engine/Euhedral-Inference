package io.euhedral_execution.inference.core.model_loader.config;

public record QwenConfig(
        int vocabSize,
        int hiddenSize,
        int numHiddenLayers,

        int numAttentionHeads,
        int numKeyValueHeads,
        int attentionHeadDim,

        int intermediateSize,

        int linearNumKeyHeads,
        int linearNumValueHeads,
        int linearKeyHeadDim,
        int linearValueHeadDim,
        int linearConvKernelDim,

        double rmsNormEpsilon,

        double ropeTheta,
        double partialRotaryFactor,
        int maxPositionEmbeddings,

        String hiddenActivation,

        QwenLayerType[] layerTypes,

        int numExperts,
        int numExpertsPerToken,
        int moeIntermediateSize,
        int sharedExpertIntermediateSize,

        boolean tieWordEmbeddings,
        boolean attentionOutputGate,

        int mtpLayerCount) {}
