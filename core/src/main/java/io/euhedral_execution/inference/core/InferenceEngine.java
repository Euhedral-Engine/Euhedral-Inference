package io.euhedral_execution.inference.core;

import io.euhedral_execution.core.config.LatticeConfig;
import io.euhedral_execution.core.control_plane.ControlPlaneLattice;
import io.euhedral_execution.core.control_plane.ControlPlaneShard;
import io.euhedral_execution.core.impl.BaseCloneableObject;
import io.euhedral_execution.core.impl.DefaultExecutor;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifact;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.scheduling.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.QwenGenerationSession;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;

/// High-level owner of model, CUDA backend, Euhedral lattice, and all sessions it creates.
/// Close sessions early when finished; engine close also closes every tracked session.
/// This standalone application's fabric is process-wide. The engine controls its start/stop;
/// independent sessions and their sources share it. Only one engine may control that lifecycle.
/// Initialize the process fabric through load, not a separate low-level lattice factory; advanced
/// sources may borrow the running fabric but must not initialize or stop it independently.
public final class InferenceEngine implements AutoCloseable {
    private static final AtomicBoolean LATTICE_OWNED = new AtomicBoolean();
    private static final AtomicReference<ControlPlaneLattice> LAST_CLOSED_LATTICE = new AtomicReference<>();
    private static final AtomicLong SEQUENCE_IDS = new AtomicLong();
    private final Bootstrap bootstrap;
    private final QwenTokenizer tokenizer;
    private final ExecutionGpu gpu;
    private final QwenModel model;
    private final ControlPlaneLattice lattice;
    private final QwenExecutionPlan plan;
    private final EuhedralInferenceRuntime runtime;
    private final List<QwenGenerationSession> sessions = new ArrayList<>();
    private final ReentrantLock shutdownLock = new ReentrantLock();
    private volatile boolean closing;
    private boolean resourcesClosed;

    private InferenceEngine(
            Bootstrap bootstrap,
            QwenTokenizer tokenizer,
            ExecutionGpu gpu,
            QwenModel model,
            ControlPlaneLattice lattice,
            QwenExecutionPlan plan,
            EuhedralInferenceRuntime runtime) {
        this.bootstrap = bootstrap;
        this.tokenizer = tokenizer;
        this.gpu = gpu;
        this.model = model;
        this.lattice = lattice;
        this.plan = plan;
        this.runtime = runtime;
    }

    /// Loads all model resources and starts the lattice before returning an engine.
    public static InferenceEngine load(InferenceConfig config) throws IOException {
        return load(config, new Bootstrap());
    }

    static InferenceEngine load(InferenceConfig config, Bootstrap bootstrap) throws IOException {
        Objects.requireNonNull(config, "config");
        if (!LATTICE_OWNED.compareAndSet(false, true))
            throw new IllegalStateException("an inference engine already owns the process-wide Euhedral lattice");
        ExecutionGpu gpu = null;
        QwenModel model = null;
        ControlPlaneLattice lattice = null;
        try {
            QwenTokenizer tokenizer = QwenTokenizer.load(config.tokenizerDirectory());
            QwenArtifact artifact = bootstrap.readArtifact(config.artifactPath());
            gpu = bootstrap.openGpu(config.cudaLibraryPath());
            model = bootstrap.loadModel(config.artifactPath(), artifact, gpu);
            QwenExecutionPlan plan = new QwenExecutionPlan(model.weights());
            lattice = bootstrap.createLattice(config);
            bootstrap.startLattice(lattice);
            EuhedralInferenceRuntime runtime = new EuhedralInferenceRuntime(lattice, plan, gpu);
            return new InferenceEngine(bootstrap, tokenizer, gpu, model, lattice, plan, runtime);
        } catch (IOException | RuntimeException | Error failure) {
            var pending = new StartupFailure(
                    failure,
                    bootstrap,
                    gpu,
                    failure instanceof QwenModel.LoadFailure partial ? partial : model,
                    lattice);
            try {
                pending.close();
            } catch (RuntimeException | Error cleanup) {
                pending.addSuppressed(cleanup);
                throw pending;
            }
            throw failure;
        }
    }

    /// Startup and its rollback both failed. This exception retains ownership of unreleased resources;
    /// close it again to retry cleanup before loading another engine. No session has been admitted.
    public static final class StartupFailure extends IOException implements AutoCloseable {
        private final Bootstrap bootstrap;
        private ExecutionGpu gpu;
        private AutoCloseable model;
        private ControlPlaneLattice lattice;
        private boolean cleaned;

        private StartupFailure(
                Throwable failure,
                Bootstrap bootstrap,
                ExecutionGpu gpu,
                AutoCloseable model,
                ControlPlaneLattice lattice) {
            super("inference startup failed; retry close on this exception to finish resource cleanup", failure);
            this.bootstrap = bootstrap;
            this.gpu = gpu;
            this.model = model;
            this.lattice = lattice;
        }

        @Override
        public synchronized void close() {
            if (this.cleaned) return;
            if (this.lattice != null) {
                LAST_CLOSED_LATTICE.set(this.lattice);
                this.lattice.close();
                this.lattice = null;
            }
            if (this.model != null) {
                try {
                    this.model.close();
                } catch (RuntimeException | Error cleanup) {
                    throw cleanup;
                } catch (Exception cleanup) {
                    throw new IllegalStateException("model cleanup failed", cleanup);
                }
                this.model = null;
            }
            if (this.gpu != null) {
                this.bootstrap.closeGpu(this.gpu);
                this.gpu = null;
            }
            this.cleaned = true;
            LATTICE_OWNED.set(false);
        }
    }

    /// Creates independent sequence/sampler/decoder state borrowing this engine's shared runtime.
    public synchronized QwenGenerationSession createSession(GenerationConfig config) {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        var session = new QwenGenerationSession(
                this.tokenizer,
                this.plan,
                this.runtime,
                this.gpu,
                SEQUENCE_IDS.getAndIncrement(),
                Objects.requireNonNull(config, "config"),
                this::releaseSession);
        this.sessions.add(session);
        return session;
    }

    private synchronized void releaseSession(QwenGenerationSession session) {
        this.sessions.remove(session);
    }

    public QwenTokenizer tokenizer() {
        return this.tokenizer;
    }

    public QwenConfig modelConfig() {
        return this.model.weights().config();
    }
    /// True as soon as shutdown begins; no further sessions can be admitted.
    public boolean isClosed() {
        return this.closing;
    }

    public synchronized CudaGpuMemory.DeviceMemoryInfo deviceMemoryInfo() {
        if (this.closing) throw new IllegalStateException("inference engine is closed");
        return this.bootstrap.memoryInfo(this.gpu);
    }

    /// Stops admission, closes sessions, detaches execution, closes the lattice, then frees model and CUDA.
    /// Do not call from a generation callback: it would wait for that same generation to finish.
    /// If cleanup throws, admission stays closed and a later close retries unreleased resources.
    @Override
    public void close() {
        List<QwenGenerationSession> owned;
        synchronized (this) {
            for (var session : this.sessions) {
                if (session.isGeneratingOnCurrentThread())
                    throw new IllegalStateException("cannot close the engine from its generation callback");
            }
            this.closing = true;
            owned = List.copyOf(this.sessions);
        }
        this.shutdownLock.lock();
        try {
            if (this.resourcesClosed) return;
            // Cancel all before waiting for any one generation. No admission monitor is held while waiting.
            Throwable failure = null;
            for (var session : owned) {
                try {
                    session.cancel();
                } catch (RuntimeException | Error cleanup) {
                    if (failure == null) failure = cleanup;
                    else suppress(failure, cleanup);
                }
            }
            for (var session : owned) {
                try {
                    session.close();
                } catch (RuntimeException | Error cleanup) {
                    if (failure == null) failure = cleanup;
                    else suppress(failure, cleanup);
                }
            }
            // Fail closed: never unload resources after an unproven session shutdown.
            if (failure instanceof RuntimeException exception) throw exception;
            if (failure instanceof Error error) throw error;
            this.runtime.disconnectRunner();
            // All inference sources have drained before fabric shutdown, even if fabric teardown is asynchronous.
            LAST_CLOSED_LATTICE.set(this.lattice);
            this.lattice.close();
            this.model.close();
            this.bootstrap.closeGpu(this.gpu);
            this.resourcesClosed = true;
            synchronized (this) {
                this.sessions.clear();
            }
            LATTICE_OWNED.set(false);
        } finally {
            this.shutdownLock.unlock();
        }
    }

    private static void suppress(Throwable failure, Throwable cleanup) {
        if (failure != cleanup) failure.addSuppressed(cleanup);
    }

    // Package-private construction seam for failure injection, not an alternative public backend API.
    static class Bootstrap {
        QwenArtifact readArtifact(Path path) throws IOException {
            return QwenArtifactReader.read(path);
        }

        ExecutionGpu openGpu(Path path) {
            return new CudaGpuMemory(path);
        }

        QwenModel loadModel(Path path, QwenArtifact artifact, ExecutionGpu gpu) throws IOException {
            return QwenModel.load(path, artifact, gpu);
        }

        ControlPlaneLattice createLattice(InferenceConfig config) {
            var shard =
                    ControlPlaneShard.createBaseShard("InferenceShard", new BaseCloneableObject(new DefaultExecutor()));
            var latticeConfig =
                    new LatticeConfig("InferenceLattice", config.workerCpus(), config.shutdownTimeout(), shard);
            long started = System.nanoTime();
            while (true) {
                ControlPlaneLattice lattice = ControlPlaneLattice.getOrCreate(latticeConfig);
                if (lattice != LAST_CLOSED_LATTICE.get()) return lattice;
                // close may return while singleton removal finishes. Never return that stopped fabric on reload.
                if (Thread.currentThread().isInterrupted()
                        || System.nanoTime() - started
                                >= config.shutdownTimeout().toNanos())
                    throw new IllegalStateException("previous Euhedral fabric has not finished shutting down");
                LockSupport.parkNanos(1_000_000L);
            }
        }

        void startLattice(ControlPlaneLattice lattice) {
            lattice.start();
        }

        void closeGpu(ExecutionGpu gpu) {
            ((CudaGpuMemory) gpu).close();
        }

        CudaGpuMemory.DeviceMemoryInfo memoryInfo(ExecutionGpu gpu) {
            return ((CudaGpuMemory) gpu).deviceMemoryInfo();
        }
    }
}
