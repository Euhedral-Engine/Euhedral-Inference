package io.euhedral_execution.inference.core;

import static org.junit.jupiter.api.Assertions.*;

import io.euhedral_execution.hardware_utils.SystemInfo;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.EngineModelFixture;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.EngineExecutionFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.BitSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

@Timeout(60)
class InferenceEngineTest {
    @TempDir
    Path directory;

    @Test
    void ownsSessionsAndClosesModelAfterThem() throws Exception {
        var bootstrap = new FakeBootstrap();
        try (var engine = InferenceEngine.load(config(), bootstrap)) {
            var first = engine.createSession(GenerationConfig.greedy(1L));
            var second = engine.createSession(GenerationConfig.greedy(1L));
            assertNotSame(first, second);
            assertNotEquals(
                    EngineExecutionFixture.sequence(first).sequenceId(),
                    EngineExecutionFixture.sequence(second).sequenceId());
            assertEquals(first.generate("!", 2, ignored -> {}), second.generate("!", 2, ignored -> {}));
            assertNotSame(
                    EngineExecutionFixture.sequence(first).kvCacheState(),
                    EngineExecutionFixture.sequence(second).kvCacheState());
            assertNotSame(
                    EngineExecutionFixture.sequence(first).recurrentState(),
                    EngineExecutionFixture.sequence(second).recurrentState());
            assertEquals(1, bootstrap.gpu.embeddingAddresses.stream().distinct().count());
            assertEquals(1, bootstrap.modelLoads);
            first.close();
            assertFalse(second.isClosed());
            assertFalse(bootstrap.gpuClosed);
            assertFalse(bootstrap.gpu.freed().contains(1000L), "model freed by session close");
            second.generate("!", 1, ignored -> {});
            engine.close();
            assertTrue(first.isClosed());
            assertTrue(second.isClosed());
            assertTrue(engine.isClosed());
            assertTrue(bootstrap.gpuClosed);
            assertTrue(bootstrap.gpu.freed().containsAll(bootstrap.gpu.allocated()));
            assertThrows(IllegalStateException.class, () -> engine.createSession(GenerationConfig.greedy(2L)));
            engine.close();
            assertEquals(1, bootstrap.gpuCloseCount);
        }
    }

    InferenceConfig config() throws Exception {
        Files.writeString(directory.resolve("tokenizer.json"), """
            {"model":{"type":"BPE","vocab":{"!":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6},"merges":[]},
             "normalizer":{"type":"NFC"},
             "pre_tokenizer":{"type":"Sequence","pretokenizers":[
             {"type":"Split","pattern":{"Regex":"."},"behavior":"Isolated","invert":false},
             {"type":"ByteLevel"}]},
             "added_tokens":[{"content":"<eos>","id":7,"special":true}]}
            """);
        Files.writeString(directory.resolve("tokenizer_config.json"), "{\"eos_token\":\"<eos>\"}");
        Files.writeString(directory.resolve("generation_config.json"), "{\"eos_token_id\":7}");
        BitSet cpus = new BitSet();
        cpus.set(SystemInfo.getPCpuSet().nextSetBit(0));
        return new InferenceConfig(
                directory.resolve("model.edrl"), directory, directory.resolve("lib.so"), cpus, Duration.ofSeconds(10));
    }

    @Test
    void concurrentSessionsShareRuntimeWithoutRunnerCollision() throws Exception {
        var bootstrap = new FakeBootstrap();
        var entered = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        bootstrap.gpu.afterEmbedding(() -> {
            entered.countDown();
            await(release);
        });
        try (var engine = InferenceEngine.load(config(), bootstrap);
                var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
            var first = engine.createSession(GenerationConfig.greedy(1L));
            var second = engine.createSession(GenerationConfig.greedy(2L));
            var a = executor.submit(() -> first.generate("!", 2, ignored -> {}));
            assertTrue(entered.await(10, java.util.concurrent.TimeUnit.SECONDS));
            var b = executor.submit(() -> second.generate("!", 3, ignored -> {}));
            try {
                // A second quantum must wait, rather than fail while the first runner is attached.
                assertThrows(
                        java.util.concurrent.TimeoutException.class,
                        () -> b.get(100, java.util.concurrent.TimeUnit.MILLISECONDS));
            } finally {
                release.countDown();
            }
            assertEquals(2, a.get(10, java.util.concurrent.TimeUnit.SECONDS).size());
            assertEquals(3, b.get(10, java.util.concurrent.TimeUnit.SECONDS).size());
            assertEquals(3, first.currentTokenPosition());
            assertEquals(4, second.currentTokenPosition());
        } finally {
            release.countDown();
        }
    }

    @Test
    void startupFailureClosesUploadedModelAndBackend() throws Exception {
        var bootstrap = new FakeBootstrap() {
            @Override
            void startLattice(io.euhedral_execution.core.control_plane.ControlPlaneLattice lattice) {
                super.startLattice(lattice);
                throw new IllegalStateException("after lattice start");
            }
        };
        assertEquals(
                "after lattice start",
                assertThrows(IllegalStateException.class, () -> InferenceEngine.load(config(), bootstrap))
                        .getMessage());
        assertTrue(bootstrap.gpuClosed);
        assertEquals(bootstrap.gpu.allocated(), bootstrap.gpu.freed());
        try (var next = InferenceEngine.load(config(), new FakeBootstrap())) {
            assertFalse(next.isClosed());
        }
    }

    @Test
    void shutdownFromCallbackIsRejectedBeforeClosingBegins() throws Exception {
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            var session = engine.createSession(GenerationConfig.greedy(1L));
            session.generate("!", 1, text -> {
                assertThrows(IllegalStateException.class, engine::close);
                assertFalse(engine.isClosed());
            });
            assertEquals(2, session.currentTokenPosition());
        }
    }

    static void await(java.util.concurrent.CountDownLatch latch) {
        try {
            if (!latch.await(10, java.util.concurrent.TimeUnit.SECONDS))
                throw new IllegalStateException("gate timeout");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    @Test
    void closingRejectsAdmissionAndQueriesBeforeWaitingForActiveExecution() throws Exception {
        var bootstrap = new FakeBootstrap();
        var entered = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        bootstrap.gpu.afterEmbedding(() -> {
            entered.countDown();
            await(release);
        });
        try (var engine = InferenceEngine.load(config(), bootstrap);
                var executor = java.util.concurrent.Executors.newFixedThreadPool(3)) {
            var session = engine.createSession(GenerationConfig.greedy(1L));
            var generation = executor.submit(() -> session.generate("!", 4, ignored -> {}));
            assertTrue(entered.await(10, java.util.concurrent.TimeUnit.SECONDS));
            var close = executor.submit(engine::close);
            long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5);
            while (!engine.isClosed() && System.nanoTime() < deadline) Thread.onSpinWait();
            try {
                assertTrue(engine.isClosed());
                assertFalse(bootstrap.gpuClosed);
                assertThrows(IllegalStateException.class, () -> engine.createSession(GenerationConfig.greedy(2L)));
                var query = executor.submit(() -> assertThrows(IllegalStateException.class, engine::deviceMemoryInfo));
                query.get(1, java.util.concurrent.TimeUnit.SECONDS);
            } finally {
                release.countDown();
            }
            generation.get(10, java.util.concurrent.TimeUnit.SECONDS);
            close.get(10, java.util.concurrent.TimeUnit.SECONDS);
            assertTrue(session.isClosed());
            assertTrue(bootstrap.gpuClosed);
            assertTrue(bootstrap.gpu.freed().containsAll(bootstrap.gpu.allocated()));
        } finally {
            release.countDown();
        }
    }

    @Test
    void failuresBeforeAndAfterGpuCreationReleaseOnlyCreatedResources() throws Exception {
        for (int stage = 0; stage < 5; stage++) {
            final int selected = stage;
            var bootstrap = new FakeBootstrap() {
                @Override
                QwenArtifact readArtifact(Path path) {
                    if (selected == 0) throw new IllegalStateException("artifact");
                    return super.readArtifact(path);
                }

                @Override
                ExecutionGpu openGpu(Path path) {
                    if (selected == 1) throw new IllegalStateException("gpu");
                    return super.openGpu(path);
                }

                @Override
                QwenModel loadModel(Path path, QwenArtifact artifact, ExecutionGpu gpu) throws java.io.IOException {
                    if (selected == 2) throw new IllegalStateException("model");
                    if (selected == 3)
                        return EngineModelFixture.load(
                                gpu,
                                new io.euhedral_execution.inference.core.model_loader.QwenWeights(
                                        null, null, null, null, null, null));
                    return super.loadModel(path, artifact, gpu);
                }

                @Override
                io.euhedral_execution.core.control_plane.ControlPlaneLattice createLattice(InferenceConfig config) {
                    throw new IllegalStateException("lattice");
                }
            };
            assertThrows(RuntimeException.class, () -> InferenceEngine.load(config(), bootstrap));
            assertEquals(selected >= 2, bootstrap.gpuClosed);
            assertEquals(bootstrap.gpu.allocated(), bootstrap.gpu.freed());
        }
        var config = config();
        Files.delete(directory.resolve("tokenizer.json"));
        var bootstrap = new FakeBootstrap();
        assertThrows(java.io.IOException.class, () -> InferenceEngine.load(config, bootstrap));
        assertFalse(bootstrap.gpuClosed);
        try (var engine = InferenceEngine.load(config(), new FakeBootstrap())) {
            assertFalse(engine.isClosed());
        }
    }

    @Test
    void rejectsASecondEngineWithoutDisturbingTheFirst() throws Exception {
        var config = config();
        try (var engine = InferenceEngine.load(config, new FakeBootstrap())) {
            var rejected = new FakeBootstrap();
            assertThrows(IllegalStateException.class, () -> InferenceEngine.load(config, rejected));
            assertFalse(rejected.gpuClosed);
            engine.createSession(GenerationConfig.greedy(1L)).generate("!", 1, ignored -> {});
        }
    }

    @Test
    void configurationCopiesCpuSetAndRejectsEmptyWorkers() throws Exception {
        var config = config();
        var returned = config.workerCpus();
        returned.clear();
        assertFalse(config.workerCpus().isEmpty());
        assertThrows(
                IllegalArgumentException.class,
                () -> new InferenceConfig(
                        config.artifactPath(),
                        directory,
                        config.cudaLibraryPath(),
                        new BitSet(),
                        Duration.ofSeconds(1)));
    }

    @Test
    void closeRetainsBackendUntilPersistentCleanupCanBeRetried() throws Exception {
        var boot = new FakeBootstrap();
        var engine = InferenceEngine.load(config(), boot);
        var session = engine.createSession(GenerationConfig.greedy(1L));
        session.generate("!", 1, ignored -> {});
        boot.gpu.freeFailures.set(100);
        try {
            assertThrows(IllegalStateException.class, engine::close);
            assertTrue(engine.isClosed());
            assertEquals(0, boot.gpuCloseCount);
            assertFalse(boot.gpu.freed().contains(1000L));
        } finally {
            boot.gpu.freeFailures.set(0);
            engine.close();
        }
        assertEquals(1, boot.gpuCloseCount);
        assertEquals(new java.util.HashSet<>(boot.gpu.allocated()), new java.util.HashSet<>(boot.gpu.freed()));
        assertEquals(boot.gpu.freed().size(), new java.util.HashSet<>(boot.gpu.freed()).size());
    }

    @Test
    void interruptedCloseDoesNotReuseAStoppedFabricOnReload() throws Exception {
        var configuration = config();
        var first = InferenceEngine.load(configuration, new FakeBootstrap());
        Thread.currentThread().interrupt();
        try {
            first.close();
        } finally {
            Thread.interrupted();
        }
        try (var next = InferenceEngine.load(configuration, new FakeBootstrap());
                var session = next.createSession(GenerationConfig.greedy(1L))) {
            assertEquals(java.util.List.of(1), session.generate("!", 1, ignored -> {}));
        }
    }

    @Test
    void startupRollbackFailureRetainsBackendAndCanBeClosedExplicitly() throws Exception {
        var boot = new FakeBootstrap() {
            @Override
            void startLattice(io.euhedral_execution.core.control_plane.ControlPlaneLattice lattice) {
                super.startLattice(lattice);
                this.gpu.freeFailures.set(100);
                throw new IllegalStateException("startup failed");
            }
        };
        var configuration = config();
        var failed =
                assertThrows(InferenceEngine.StartupFailure.class, () -> InferenceEngine.load(configuration, boot));
        try {
            assertEquals("startup failed", failed.getCause().getMessage());
            assertFalse(boot.gpuClosed);
            assertFalse(boot.gpu.freed().contains(1000L));
            assertThrows(IllegalStateException.class, () -> InferenceEngine.load(configuration, new FakeBootstrap()));
        } finally {
            boot.gpu.freeFailures.set(0);
            failed.close();
        }
        failed.close();
        assertEquals(1, boot.gpuCloseCount);
        assertEquals(boot.gpu.allocated(), boot.gpu.freed());
        try (var next = InferenceEngine.load(configuration, new FakeBootstrap())) {
            assertFalse(next.isClosed());
        }
    }

    static class FakeBootstrap extends InferenceEngine.Bootstrap {
        final EngineExecutionFixture.SamplingGpu gpu = new EngineExecutionFixture.SamplingGpu(8);
        volatile boolean gpuClosed;
        int gpuCloseCount;
        int modelLoads;

        @Override
        QwenArtifact readArtifact(Path path) {
            return null;
        }

        @Override
        ExecutionGpu openGpu(Path path) {
            return gpu;
        }

        @Override
        QwenModel loadModel(Path path, QwenArtifact artifact, ExecutionGpu gpu) throws java.io.IOException {
            modelLoads++;
            return EngineModelFixture.load(gpu, EngineExecutionFixture.weights());
        }

        @Override
        void closeGpu(ExecutionGpu gpu) {
            gpuClosed = true;
            gpuCloseCount++;
        }
    }
}
