package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.engine.InferenceBackend;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/// Liveness and engine readiness outside `/v1`. Reports only the public model ID.
///
/// The application cannot serve HTTP before the engine is initialized, so a reachable endpoint implies
/// the engine loaded; `engine` turns to `closed` (with 503) once shutdown begins.
@RestController
public class HealthController {
    private final InferenceBackend backend;

    public HealthController(InferenceBackend backend) {
        this.backend = backend;
    }

    public record Health(String status, String engine, String model) {}

    @GetMapping(path = "/health", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Health> health() {
        if (this.backend.isAvailable()) return ResponseEntity.ok(new Health("up", "ready", this.backend.modelId()));
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(new Health("up", "closed", this.backend.modelId()));
    }
}
