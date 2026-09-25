package io.euhedral_execution.inference.benchmark.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.benchmark.BenchmarkFixtures;
import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.measure.IterationTiming;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import io.euhedral_execution.inference.benchmark.result.Summary;
import io.euhedral_execution.inference.core.InferenceTuning;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;

class BenchmarkRunnerTest {
    @Test
    void warmupRowsPrecedeMeasuredRowsWithFreshSessionsPerIterationAndEnginePerChunk() throws Exception {
        List<Scenario> scenarios =
                List.of(new Scenario(Scenario.Kind.PREFILL, 10, 0), new Scenario(Scenario.Kind.SUSTAINED_DECODE, 3, 4));
        var options = BenchmarkFixtures.options(List.of(4, 2), scenarios, 2, 3);
        List<BenchmarkFixtures.FakeTarget> targets = new ArrayList<>();
        List<BenchmarkResult> rows = new ArrayList<>();
        BenchmarkRunner.run(
                options,
                InferenceTuning.defaults(BenchmarkFixtures.bits(0)),
                BenchmarkFixtures.prompts(scenarios),
                tuning -> {
                    var target = new BenchmarkFixtures.FakeTarget(tuning);
                    targets.add(target);
                    return target;
                },
                BenchmarkFixtures.context(),
                rows::add);

        assertEquals(2, targets.size(), "one engine per prefill-chunk value");
        assertTrue(targets.stream().allMatch(target -> target.closed));
        assertEquals(10, targets.getFirst().sessions, "2 scenarios x (2 warmup + 3 measured)");
        var distinct = new IdentityHashMap<IterationTiming, Boolean>();
        targets.forEach(target -> target.timings.forEach(timing -> distinct.put(timing, true)));
        assertEquals(20, distinct.size(), "every iteration has its own timing and session");
        assertEquals(20, rows.size());

        List<String> order = rows.subList(0, 5).stream()
                .map(row -> (row.warmup() ? "w" : "m") + row.iteration())
                .toList();
        assertEquals(List.of("w0", "w1", "m0", "m1", "m2"), order);
        assertEquals(
                4,
                rows.getFirst()
                        .engine()
                        .path("tuning")
                        .path("prefillChunkTokens")
                        .asInt());
        assertEquals(
                2,
                rows.getLast()
                        .engine()
                        .path("tuning")
                        .path("prefillChunkTokens")
                        .asInt());
        var prefill = rows.get(2);
        assertEquals("prefill-10", prefill.scenario().name());
        assertEquals(BenchmarkResult.SUCCESS, prefill.status());
        assertEquals(3, prefill.work().prefillQuanta(), "ten prompt tokens in chunks of four");
        assertEquals(10, prefill.work().promptTokens());
        assertEquals(PromptMaterial.GENERATOR, prefill.scenario().promptGenerator());
        var decode = rows.get(7);
        assertEquals("decode-3-4", decode.scenario().name());
        assertEquals(4, decode.work().generatedTokens());
        assertEquals(3, decode.work().decodeSampledTokens());
        assertEquals(1, decode.work().finalCommitQuanta());
        assertEquals(1_000L, decode.gpuMemory().totalBytes());
        assertEquals("unspecified", decode.fork().source());
        assertTrue(Summary.format(rows).contains("euhedral-inference | chunk 4 | decode-3-4"));
    }

    @Test
    void failedAndEosIterationsAreRecordedWithoutStoppingTheRun() throws Exception {
        List<Scenario> scenarios = List.of(new Scenario(Scenario.Kind.SUSTAINED_DECODE, 2, 8));
        var options = BenchmarkFixtures.options(List.of(512), scenarios, 0, 3);
        List<BenchmarkResult> rows = new ArrayList<>();
        BenchmarkRunner.run(
                options,
                InferenceTuning.defaults(BenchmarkFixtures.bits(0)),
                BenchmarkFixtures.prompts(scenarios),
                tuning -> {
                    var target = new BenchmarkFixtures.FakeTarget(tuning);
                    target.failOnCall = 1;
                    target.eosAfter = 3;
                    return target;
                },
                BenchmarkFixtures.context(),
                rows::add);
        assertEquals(
                List.of(BenchmarkResult.INELIGIBLE, BenchmarkResult.FAILED, BenchmarkResult.INELIGIBLE),
                rows.stream().map(BenchmarkResult::status).toList());
        assertEquals(3, rows.getFirst().work().generatedTokens(), "actual, not requested, token count");
        assertEquals("eos_after_3_of_8_tokens", rows.getFirst().statusReason());
        assertNull(rows.get(1).throughput());
        assertEquals(
                "IllegalStateException: injected quantum failure", rows.get(1).statusReason());
        String summary = Summary.format(rows);
        assertTrue(summary.contains("   0    2    1"), summary);
    }

    @Test
    void summaryIgnoresWarmupRows() throws Exception {
        List<Scenario> scenarios = List.of(new Scenario(Scenario.Kind.PREFILL, 4, 0));
        var options = BenchmarkFixtures.options(List.of(512), scenarios, 1, 1);
        List<BenchmarkResult> rows = new ArrayList<>();
        BenchmarkRunner.run(
                options,
                InferenceTuning.defaults(BenchmarkFixtures.bits(0)),
                BenchmarkFixtures.prompts(scenarios),
                BenchmarkFixtures.FakeTarget::new,
                BenchmarkFixtures.context(),
                rows::add);
        assertTrue(rows.getFirst().warmup());
        assertFalse(rows.getLast().warmup());
        assertNotSame(rows.getFirst(), rows.getLast());
        assertTrue(Summary.format(rows).contains("   1    0    0"), Summary.format(rows));
    }
}
