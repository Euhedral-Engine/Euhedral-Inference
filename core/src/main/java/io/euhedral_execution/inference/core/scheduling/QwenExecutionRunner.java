package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.data_structures.queues.PartitionedMpscQueue;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;

/// A directly pluggable Euhedral source. Producers publish ready operation state to an MPSC queue;
/// Euhedral owns `pull`, `request`, frame checkout, and synchronous pushes.
public final class QwenExecutionRunner implements LatticeSource {

    private static final Function<AbstractFrame, Boolean> NEVER_STOP = ignored -> false;

    private final QwenExecutionPlan plan;
    private final ExecutionGpu gpu;
    private final PartitionedMpscQueue<QwenExecutionContext> ready;
    private final QwenWorkGenerator generator;
    private final AtomicInteger active = new AtomicInteger();
    private final AtomicBoolean attached = new AtomicBoolean();

    private final AtomicReference<LatticeReceiver> downstream = new AtomicReference<>();
    private final AtomicBoolean finished = new AtomicBoolean();

    public QwenExecutionRunner(QwenExecutionPlan plan, ExecutionGpu gpu) {
        this(plan, gpu, ignored -> {});
    }

    public QwenExecutionRunner(
            QwenExecutionPlan plan, ExecutionGpu gpu, Consumer<? super QwenExecutionContext> terminalConsumer) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.ready = new PartitionedMpscQueue<>(plan.instructions().size(), 64);
        this.generator = new QwenWorkGenerator(plan, gpu, this, terminalConsumer);
    }

    public CompletableFuture<QwenExecutionContext.Outcome> submit(QwenExecutionContext context) {
        Objects.requireNonNull(context, "context");
        if (context.plan() != plan) {
            throw new IllegalArgumentException("quantum belongs to another execution plan");
        }
        while (true) {
            int count = active.get();
            if (count < 0) throw new IllegalStateException("Qwen runner admission is closed");
            if (count == Integer.MAX_VALUE) throw new IllegalStateException("too many active quanta");
            if (active.compareAndSet(count, count + 1)) break;
        }
        try {
            context.begin(gpu);
        } catch (RuntimeException | Error failure) {
            if (active.decrementAndGet() == Integer.MIN_VALUE) signalComplete();
            throw failure;
        }
        var outcome = context.completion();
        outcome.whenComplete((ignored, failure) -> {
            int remaining = active.decrementAndGet();
            if (remaining == Integer.MIN_VALUE) signalComplete();
        });
        if (!outcome.isDone()) generator.start(context);
        return outcome.copy();
    }

    /// Called by producers; each partition is bound to one immutable plan instruction.
    boolean offerReady(int instructionId, QwenExecutionContext context) {
        return this.ready.offer(instructionId, context);
    }

    QwenExecutionContext peekReady(int instructionId) {
        return this.ready.peek(instructionId);
    }

    QwenExecutionContext pollReady(int instructionId) {
        return this.ready.poll(instructionId);
    }

    @Override
    public void addDownstream(LatticeReceiver receiver) {
        Objects.requireNonNull(receiver, "receiver");
        if (!attached.compareAndSet(false, true)) {
            receiver.onError(new IllegalStateException("Qwen source already has a downstream"));
        } else if (finished.get()) {
            receiver.onComplete();
        } else {
            downstream.set(receiver);
            if (finished.get() && downstream.compareAndSet(receiver, null)) receiver.onComplete();
        }
    }

    @Override
    public long pull(Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long requested) {
        Objects.requireNonNull(consumer, "consumer");
        Objects.requireNonNull(stopCondition, "stopCondition");
        if (requested <= 0 || finished.get()) return 0;
        return this.generator.drain(consumer, stopCondition, requested);
    }

    @Override
    public void request(long requested) {
        if (requested <= 0 || finished.get()) return;
        LatticeReceiver receiver = downstream.get();
        if (receiver != null) this.generator.drain(receiver::push, NEVER_STOP, requested);
    }

    @Override
    public void complete() {
        completeGracefully();
    }

    public void completeGracefully() {
        int old = active.getAndUpdate(value -> value | Integer.MIN_VALUE);
        if ((old & Integer.MAX_VALUE) == 0) signalComplete();
    }

    private void signalComplete() {
        if (finished.compareAndSet(false, true)) {
            LatticeReceiver receiver = downstream.getAndSet(null);
            if (receiver != null) receiver.onComplete();
        }
    }

    @Override
    public boolean isComplete() {
        return finished.get();
    }
}
