package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.frames.RunnableFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.QueueIngestSink;
import io.euhedral_execution.data_structures.queues.PartitionedMpscQueue;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class CudaAsyncCompletionIntegrationTest {
    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void partialWorkerStartupReleasesOrphanedStreams() {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "a candidate CUDA library is required");
        try (var gpu = new CudaGpuMemory(Path.of(selected), true)) {
            gpu.openWorker(3);
            gpu.openWorker(7);
            gpu.abortWorkerStartup();
            gpu.ensureWorkersClosed();
            assertThrows(IllegalStateException.class, () -> gpu.openWorker(9));
        }
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void separateWorkersRetainIndependentStreamsAcrossFrames() throws Exception {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "a candidate CUDA library is required");
        try (var gpu = new CudaGpuMemory(Path.of(selected), true)) {
            long firstWorker = gpu.openWorker(3);
            long secondWorker = gpu.openWorker(7);
            assertThrows(IllegalStateException.class, gpu::ensureWorkersClosed);
            long first = gpu.workerStream(firstWorker);
            long second = gpu.workerStream(secondWorker);
            assertTrue(first != 0 && second != 0 && first != second);
            assertEquals(
                    first, gpu.workerStream(firstWorker), "a worker must reuse its stream for its entire lifetime");
            long replacement = gpu.openWorker(3);
            assertTrue(first != gpu.workerStream(replacement), "a replacement on the same CPU needs its own stream");
            gpu.closeWorker(firstWorker);
            assertEquals(second, gpu.workerStream(secondWorker));
            assertThrows(IllegalStateException.class, gpu::ensureWorkersClosed);
            gpu.closeWorker(secondWorker);
            gpu.closeWorker(replacement);
            gpu.ensureWorkersClosed();
            gpu.close();
            assertThrows(IllegalStateException.class, () -> gpu.openWorker(9));
        }
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void deviceAllocationDoesNotWaitForEarlierGpuWork() throws Exception {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "a candidate CUDA library is required");
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var completionFrame = new AtomicReference<Runnable>();
        try (var gpu = new CudaGpuMemory(Path.of(selected), true);
                var worker = Executors.newSingleThreadExecutor()) {
            gpu.bindCompletionSink(completion -> {
                entered.countDown();
                try {
                    if (!release.await(10, TimeUnit.SECONDS)) throw new AssertionError("callback gate timed out");
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(failure);
                }
                completionFrame.set(completion);
            });
            long source = gpu.allocate(64);
            gpu.submit(() -> gpu.zeroDeviceMemory(source, 64));
            gpu.deferCompletion(() -> {}, error -> {
                throw new AssertionError(error);
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            var allocation = worker.submit(() -> gpu.allocate(64));
            try {
                allocation.get(2, TimeUnit.SECONDS);
            } finally {
                release.countDown();
                long allocated = allocation.get(5, TimeUnit.SECONDS);
                gpu.synchronize();
                completionFrame.get().run();
                gpu.free(allocated);
                gpu.free(source);
            }
        }
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void selectedStreamUploadDoesNotWaitForEarlierGpuWork() throws Exception {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "a candidate CUDA library is required");
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var completionFrame = new AtomicReference<Runnable>();
        try (var gpu = new CudaGpuMemory(Path.of(selected), true);
                var worker = Executors.newSingleThreadExecutor()) {
            gpu.bindCompletionSink(completion -> {
                entered.countDown();
                try {
                    if (!release.await(10, TimeUnit.SECONDS)) throw new AssertionError("callback gate timed out");
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(failure);
                }
                completionFrame.set(completion);
            });
            long destination = gpu.allocate(64);
            gpu.submit(() -> gpu.zeroDeviceMemory(destination, 64));
            gpu.deferCompletion(() -> {}, error -> {
                throw new AssertionError(error);
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            var upload = gpu.allocateUploadBuffer(64);
            upload.segment().fill((byte) 0x5a);
            var submitted = worker.submit(() -> gpu.submit(() -> gpu.copyUploadToDevice(destination, upload)));
            try {
                submitted.get(2, TimeUnit.SECONDS);
            } finally {
                release.countDown();
                submitted.get(5, TimeUnit.SECONDS);
                gpu.synchronize();
                completionFrame.get().run();
                upload.close();
                gpu.free(destination);
            }
        }
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void selectedStreamDeviceCopyDoesNotWaitForEarlierGpuWork() throws Exception {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "a candidate CUDA library is required");
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var completionFrame = new AtomicReference<Runnable>();
        try (var gpu = new CudaGpuMemory(Path.of(selected), true);
                var worker = Executors.newSingleThreadExecutor()) {
            gpu.bindCompletionSink(completion -> {
                entered.countDown();
                try {
                    if (!release.await(10, TimeUnit.SECONDS)) throw new AssertionError("callback gate timed out");
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(failure);
                }
                completionFrame.set(completion);
            });
            long source = gpu.allocate(64);
            long destination = gpu.allocate(64);
            gpu.submit(() -> gpu.zeroDeviceMemory(source, 64));
            gpu.deferCompletion(() -> {}, error -> {
                throw new AssertionError(error);
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            var submitted = worker.submit(() -> gpu.submit(() -> gpu.copyDeviceToDevice(destination, source, 64)));
            try {
                submitted.get(2, TimeUnit.SECONDS);
            } finally {
                release.countDown();
                submitted.get(5, TimeUnit.SECONDS);
                gpu.synchronize();
                completionFrame.get().run();
                gpu.free(destination);
                gpu.free(source);
            }
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void selectedStreamCopiesPreserveUploadAndKvGrowthOrdering() {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "a candidate CUDA library is required");
        try (var gpu = new CudaGpuMemory(Path.of(selected), true);
                var arena = Arena.ofConfined()) {
            long bytes = 8L << 20;
            var host = arena.allocate(bytes);
            host.fill((byte) 0x5a);
            var readback = arena.allocate(64);
            long source = gpu.allocate(bytes);
            long destination = gpu.allocate(bytes);
            try {
                for (int iteration = 0; iteration < 8; iteration++) {
                    gpu.submit(() -> {
                        gpu.zeroDeviceMemory(source, bytes);
                        gpu.copyHostToDevice(source, host, bytes);
                        gpu.copyDeviceToDevice(destination, source, bytes);
                    });
                    gpu.synchronize();
                    gpu.copyDeviceToHost(readback, destination + bytes - 64, 64);
                    byte[] actual = readback.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                    for (byte value : actual) assertEquals((byte) 0x5a, value);
                }
            } finally {
                gpu.free(destination);
                gpu.free(source);
            }
        }
    }

    @Test
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void nativeStreamCallbackPublishesCompletedWorkForLatticeFinalization() throws Exception {
        String selected =
                System.getProperty("euhedral.cuda.async.library", System.getProperty("euhedral.cuda.library"));
        assumeTrue(selected != null, "pass -Peuhedral.cuda.async.library=<candidate library>");
        Path library = Path.of(selected);
        assertTrue(Files.isRegularFile(library), "missing candidate CUDA library: " + library);
        var sink = new QueueIngestSink(new PartitionedMpscQueue<>(64));
        var source = sink.getDelegate();
        source.addDownstream(new LatticeReceiver() {
            @Override
            public void addUpstream(LatticeSource upstream) {}

            @Override
            public void push(AbstractFrame frame) {
                throw new AssertionError("unexpected push");
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable error) {
                throw new AssertionError(error);
            }
        });
        var arrived = new CountDownLatch(1);
        var finished = new AtomicBoolean();
        var failure = new AtomicReference<Throwable>();
        try (var gpu = new CudaGpuMemory(library, true);
                var arena = Arena.ofConfined()) {
            gpu.bindCompletionSink(completion -> {
                assertTrue(sink.offer(new RunnableFrame(0L, completion)));
                arrived.countDown();
            });
            long address = gpu.allocate(64);
            try {
                gpu.submit(() -> gpu.zeroDeviceMemory(address, 64));
                gpu.deferCompletion(() -> finished.set(true), failure::set);
                assertFalse(finished.get());
                assertTrue(arrived.await(10, TimeUnit.SECONDS), "CUDA did not notify the completion source");
                assertFalse(finished.get(), "native callback must not finalize the frame");
                assertEquals(1, source.pull(AbstractFrame::execute, ignored -> false, 1));
                assertTrue(finished.get());
                assertEquals(null, failure.get());
                var bytes = arena.allocate(64);
                gpu.copyDeviceToHost(bytes, address, 64);
                assertArrayEquals(new byte[64], bytes.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
            } finally {
                gpu.free(address);
            }
        } finally {
            sink.complete();
        }
    }
}
