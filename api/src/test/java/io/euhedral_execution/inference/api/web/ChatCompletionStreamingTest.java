package io.euhedral_execution.inference.api.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/// Streaming and disconnect behavior against a real Tomcat connector, without CUDA.
@SpringBootTest(classes = ScriptedApiApplication.class, webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Timeout(60)
class ChatCompletionStreamingTest {
    private static final String MODEL = ScriptedInferenceBackend.MODEL_ID;
    private static final JsonMapper JSON = JsonMapper.shared();

    @LocalServerPort
    private int port;

    @Autowired
    private ScriptedInferenceBackend backend;

    @BeforeEach
    void resetBackend() {
        this.backend.reset();
    }

    @Test
    void streamEmitsFramedChunksIncrementallyThenDone() throws Exception {
        var firstChunkSeen = new CountDownLatch(1);
        this.backend.script = (generation, max, output) -> {
            generation.emit(output, "Hel");
            // Generation cannot continue until the client has received "Hel": proves nothing is buffered.
            if (!firstChunkSeen.await(10, TimeUnit.SECONDS))
                throw new ExecutionException(new AssertionError("first chunk was not delivered while generating"));
            generation.emit(output, "lo");
            return new InferenceBackend.Result(3, true);
        };
        var request = HttpRequest.newBuilder(URI.create("http://localhost:" + this.port + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"model\":\"" + MODEL + "\",\"stream\":true,"
                        + "\"stream_options\":{\"include_usage\":true},"
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"))
                .build();
        HttpResponse<java.util.stream.Stream<String>> response;
        try (var client = HttpClient.newHttpClient()) {
            response = client.send(request, HttpResponse.BodyHandlers.ofLines());
            assertEquals(200, response.statusCode());
            assertTrue(
                    response.headers().firstValue("Content-Type").orElseThrow().startsWith("text/event-stream"));

            List<String> data = new ArrayList<>();
            var lines = response.body().iterator();
            boolean expectData = true;
            while (lines.hasNext()) {
                String line = lines.next();
                if (expectData) {
                    assertTrue(line.startsWith("data:"), "each event must be a single data field: " + line);
                    data.add(line.substring("data:".length()).strip());
                    if (data.getLast().contains("\"Hel\"")) firstChunkSeen.countDown();
                } else {
                    assertEquals("", line, "events must be separated by a blank line");
                }
                expectData = !expectData;
            }

            assertEquals("[DONE]", data.getLast());
            List<JsonNode> chunks = new ArrayList<>();
            for (String event : data.subList(0, data.size() - 1)) chunks.add(JSON.readTree(event));
            assertEquals(5, chunks.size(), data.toString());
            String id = chunks.getFirst().get("id").asString();
            for (JsonNode chunk : chunks) {
                assertEquals(id, chunk.get("id").asString());
                assertEquals("chat.completion.chunk", chunk.get("object").asString());
                assertEquals(MODEL, chunk.get("model").asString());
            }
            JsonNode role = chunks.get(0).get("choices").get(0);
            assertEquals("assistant", role.get("delta").get("role").asString());
            assertTrue(role.get("finish_reason").isNull());
            assertEquals(
                    "Hel",
                    chunks.get(1)
                            .get("choices")
                            .get(0)
                            .get("delta")
                            .get("content")
                            .asString());
            assertEquals(
                    "lo",
                    chunks.get(2)
                            .get("choices")
                            .get(0)
                            .get("delta")
                            .get("content")
                            .asString());
            JsonNode terminal = chunks.get(3).get("choices").get(0);
            assertEquals("stop", terminal.get("finish_reason").asString());
            assertTrue(terminal.get("delta").isEmpty());
            JsonNode usage = chunks.get(4);
            assertTrue(usage.get("choices").isEmpty());
            assertEquals(3, usage.get("usage").get("completion_tokens").asInt());
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed());
        assertEquals(1, generation.closeCount.get());
    }

    @Test
    void streamingClientAbortCancelsAndClosesTheSession() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(20);
        try (var client = SseTestClient.post(
                this.port,
                "/v1/chat/completions",
                "{\"model\":\"" + MODEL
                        + "\",\"stream\":true,\"max_tokens\":3000,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}")) {
            client.readUntil(line -> line.contains("\"tok1 \""));
        }
        var generation = this.backend.only();
        assertTrue(generation.awaitClosed(), "session must be closed after the client disconnects");
        assertTrue(generation.isCancelled(), "disconnect must cancel generation");
        int emitted = generation.emitted.get();
        assertTrue(emitted < 500, "generation kept running after disconnect: " + emitted + " tokens");
        Thread.sleep(200);
        assertEquals(emitted, generation.emitted.get(), "no quantum may run after the session closed");
        assertEquals(1, generation.closeCount.get());
    }

    /// Tomcat does not watch an idle async connection, so a JSON-mode disconnect surfaces only when the
    /// response is written. The session must still be closed exactly once when generation ends.
    @Test
    void nonStreamingClientAbortStillClosesTheSession() throws Exception {
        this.backend.script = ScriptedInferenceBackend.endless(10);
        ScriptedInferenceBackend.ScriptedGeneration generation;
        try (var client = SseTestClient.post(
                this.port,
                "/v1/chat/completions",
                "{\"model\":\"" + MODEL
                        + "\",\"max_tokens\":50,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}")) {
            generation = awaitGeneration();
            assertTrue(generation.started.await(10, TimeUnit.SECONDS));
        }
        assertTrue(generation.awaitClosed(), "session must be closed after an abandoned request finishes");
        assertEquals(1, generation.closeCount.get());
    }

    private ScriptedInferenceBackend.ScriptedGeneration awaitGeneration() throws InterruptedException {
        long deadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (this.backend.generations.isEmpty() && System.nanoTime() < deadline) Thread.sleep(5);
        return this.backend.only();
    }
}
