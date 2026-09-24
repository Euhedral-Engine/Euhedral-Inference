package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.chat.ChatCompletionPlan;
import io.euhedral_execution.inference.api.chat.ChatCompletionService;
import io.euhedral_execution.inference.api.chat.ChatRequestMapper;
import io.euhedral_execution.inference.api.engine.InferenceBackend;
import io.euhedral_execution.inference.api.openai.ChatCompletionRequest;
import io.euhedral_execution.inference.api.openai.ModelList;
import io.euhedral_execution.inference.api.openai.OpenAiException;
import jakarta.servlet.http.HttpServletResponse;
import java.time.Instant;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/// OpenAI-compatible surface rooted at `/v1`, so clients use `http://host:port/v1` as their base URL.
///
/// All `/v1` routes live here; authentication can later be added as a filter on `/v1/**` without
/// touching generation code.
@RestController
@RequestMapping("/v1")
public class OpenAiController {
    private final InferenceBackend backend;
    private final ChatRequestMapper requestMapper;
    private final ChatCompletionService completions;
    private final long createdAt = Instant.now().getEpochSecond();

    public OpenAiController(
            InferenceBackend backend, ChatRequestMapper requestMapper, ChatCompletionService completions) {
        this.backend = backend;
        this.requestMapper = requestMapper;
        this.completions = completions;
    }

    @GetMapping("/models")
    public ModelList models() {
        return ModelList.of(ModelList.Model.of(this.backend.modelId(), this.createdAt));
    }

    @GetMapping("/models/{model}")
    public ModelList.Model model(@PathVariable String model) {
        if (!model.equals(this.backend.modelId())) throw OpenAiException.modelNotFound(model);
        return ModelList.Model.of(model, this.createdAt);
    }

    /// Returns a `DeferredResult` for JSON or an `SseEmitter` for `stream=true`; both release this thread.
    /// Validation errors are raised here, before any response byte is committed.
    @PostMapping(path = "/chat/completions", consumes = MediaType.APPLICATION_JSON_VALUE)
    public Object chatCompletions(@RequestBody ChatCompletionRequest request, HttpServletResponse response) {
        ChatCompletionPlan plan = this.requestMapper.plan(request);
        if (!plan.stream()) return this.completions.complete(plan);
        response.setHeader("Cache-Control", "no-cache");
        // Disables proxy buffering (nginx) so chunks reach the client as they are produced.
        response.setHeader("X-Accel-Buffering", "no");
        return this.completions.stream(plan);
    }
}
