package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class InferenceEngineCudaIntegrationTest {
    @Test
    @Timeout(1800)
    void highLevelGenerationRestoresSessionAndEngineDeviceMemory() throws Exception {
        String library = System.getProperty("euhedral.cuda.library");
        assumeTrue(library != null && Files.isRegularFile(Path.of(library)));
        Path artifact = Path.of(System.getProperty(
                "euhedral.qwen.artifact", "/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl"));
        Path tokenizer =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(artifact) && Files.isRegularFile(tokenizer.resolve("tokenizer.json")));
        BitSet cpus = new BitSet();
        var available = SystemInfo.getPCpuSet();
        for (int cpu = available.nextSetBit(0); cpu >= 0 && cpus.cardinality() < 2; cpu = available.nextSetBit(cpu + 1))
            cpus.set(cpu);
        assumeTrue(cpus.cardinality() == 2);
        var config = new InferenceConfig(artifact, tokenizer, Path.of(library), cpus, Duration.ofSeconds(10));
        // This handle observes VRAM only; inference uses exclusively the public engine/session surface.
        try (var observer = new CudaGpuMemory(Path.of(library))) {
            long before = observer.deviceMemoryInfo().freeBytes();
            try (InferenceEngine engine = InferenceEngine.load(config)) {
                long loaded = engine.deviceMemoryInfo().freeBytes();
                assertTrue(before - loaded > (1L << 30), "real model was not resident");
                StringBuilder output = new StringBuilder();
                try (QwenGenerationSession session = engine.createSession(GenerationConfig.greedy(91L))) {
                    var tokens = session.generate("The capital of France is", 5, output::append);
                    assertEquals(5, tokens.size(), "expected multiple real decode quanta for the fixed prompt");
                    assertFalse(output.isEmpty());
                    var visible = tokens.stream()
                            .filter(id -> !engine.tokenizer().isGenerationEosToken(id))
                            .mapToInt(Integer::intValue)
                            .toArray();
                    assertEquals(engine.tokenizer().decode(visible), output.toString());
                    assertEquals(
                            engine.tokenizer().encodeWithModelSpecialTokens("The capital of France is").length
                                    + visible.length,
                            session.currentTokenPosition());
                    assertTrue(engine.deviceMemoryInfo().freeBytes() < loaded, "sequence did not retain device state");
                }
                long afterSession = engine.deviceMemoryInfo().freeBytes();
                assertTrue(afterSession >= loaded - (16L << 20), "session VRAM was not restored");
                System.out.println("Generated text: " + output);
                System.out.println(
                        "VRAM bytes: before=" + before + ", loaded=" + loaded + ", afterSession=" + afterSession);
            }
            long afterEngine = observer.deviceMemoryInfo().freeBytes();
            assertTrue(afterEngine >= before - (16L << 20), "engine VRAM was not restored");
            System.out.println("VRAM bytes afterEngine=" + afterEngine);
        }
    }
}
