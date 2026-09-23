package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class QwenSequenceStateTest {
    @Test
    void stateIsLockFreeAndLeaseGuardsMutation() {
        for (var method : QwenSequenceState.class.getDeclaredMethods()) {
            assertFalse(Modifier.isSynchronized(method.getModifiers()));
        }
        var state = new QwenSequenceState(21);
        var lease = state.claimExecution(0);
        assertThrows(IllegalStateException.class, () -> state.claimExecution(0));
        state.setKvCacheState(lease, "kv");
        state.setRecurrentState(lease, "recurrent");
        assertEquals("kv", state.kvCacheState());
        assertEquals("recurrent", state.recurrentState());
        state.releaseExecution(lease, 1);
        assertEquals(1, state.currentTokenPosition());
        assertThrows(IllegalStateException.class, () -> state.releaseExecution(lease, 2));
    }

    @Test
    void cancellationRacingWithClaimCannotCommitPosition() throws Exception {
        var state = new QwenSequenceState(22);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var claim = executor.submit(() -> {
                start.await();
                try {
                    return state.claimExecution(0);
                } catch (IllegalStateException cancelled) {
                    return null;
                }
            });
            var cancel = executor.submit(() -> {
                start.await();
                state.cancel();
                return null;
            });
            start.countDown();
            var lease = claim.get(5, TimeUnit.SECONDS);
            cancel.get(5, TimeUnit.SECONDS);
            if (lease != null) assertTrue(state.releaseExecutionAndCheckCancellation(lease, 1));
            assertEquals(QwenSequenceState.TerminalState.CANCELLED, state.terminalState());
            assertEquals(0, state.currentTokenPosition());
            assertFalse(state.isExecutionClaimed());
        }
    }

    @Test
    void cancellationBeforeClaimCannotOverwriteEarlierFailure() {
        var state = new QwenSequenceState(24);
        var lease = state.claimExecution(0);
        state.cancel();
        var failure = new IllegalStateException("prior quantum failed");
        state.markFailed(lease, failure);
        state.markCancelledBeforeClaim();
        assertEquals(QwenSequenceState.TerminalState.FAILED, state.terminalState());
        assertEquals(failure, state.terminalFailure());
    }

    @Test
    void concurrentClaimersPublishExactlyOneLease() throws Exception {
        var state = new QwenSequenceState(23);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(16)) {
            List<java.util.concurrent.Future<QwenSequenceState.ExecutionLease>> claims = new ArrayList<>();
            for (int index = 0; index < 16; index++) {
                claims.add(executor.submit(() -> {
                    start.await();
                    try {
                        return state.claimExecution(0);
                    } catch (IllegalStateException alreadyClaimed) {
                        return null;
                    }
                }));
            }
            start.countDown();
            int winners = 0;
            QwenSequenceState.ExecutionLease winner = null;
            for (var claim : claims) {
                var lease = claim.get(5, TimeUnit.SECONDS);
                if (lease != null) {
                    winners++;
                    winner = lease;
                }
            }
            assertEquals(1, winners);
            state.releaseExecution(winner, 1);
        }
    }
}
