package io.euhedral_execution.inference.core.tokenizer;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class JsonEnvelopeConstraintTest {
    private static QwenTokenizer tokenizer;

    @BeforeAll
    static void loadCheckpointTokenizer() throws Exception {
        Path checkpoint =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
        assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")));
        tokenizer = QwenTokenizer.load(checkpoint);
    }

    @Test
    void acceptsACompleteCallWithNestedJsonArgumentsAndThenOnlyEos() {
        var constraint = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), true);
        accept(
                constraint,
                "{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{\"path\":\"/a\",\"options\":[true,null,2.5,{\"x\":-1e3}]}}]}");
        assertTrue(constraint.complete());
        assertTrue(constraint.allows(tokenizer.eosTokenId()));
        assertFalse(constraint.allows(tokenizer.encodeText("extra")[0]));
    }

    @Test
    void autoChoiceAllowsAJsonContentAnswerWithEscapesAndUtf8() {
        var constraint = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), false);
        accept(constraint, "{\"content\":\"Line one\\nLine two 😀\"}");
        assertTrue(constraint.complete());
        assertTrue(constraint.allows(tokenizer.eosTokenId()));
    }

    @Test
    void acceptsNestedJsonArgumentsPastTheFormerFixedDepth() {
        var constraint = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), true);
        String nested = "[".repeat(65) + "0" + "]".repeat(65);
        accept(constraint, "{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{\"nested\":" + nested + "}}]}");
        assertTrue(constraint.complete());
    }

    @Test
    void nonparallelCallsCannotStartASecondCall() {
        var constraint = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), true, false);
        accept(constraint, "{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{\"path\":\"/a\"}}");
        assertFalse(constraint.allows(tokenizer.encodeText(",")[0]));
        accept(constraint, "]}");
        assertTrue(constraint.complete());

        var parallel = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), true, true);
        accept(
                parallel,
                "{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{}},"
                        + "{\"name\":\"read_file\",\"arguments\":{}}]}");
        assertTrue(parallel.complete());
    }

    @Test
    void rejectsXmlUnknownToolsIncompleteEnvelopesAndInvalidJson() {
        assertFalse(accepts("<tool_call>", false));
        assertFalse(accepts("{\"tool_calls\":[{\"name\":\"not_offered\",\"arguments\":{}}]}", false));
        assertFalse(accepts("{\"content\":\"answer\"}", true));
        assertFalse(accepts("{\"content\":\"invalid\\z\"}", false));
        assertFalse(accepts("{\"content\":\"incomplete\\u12\"}", false));
        assertFalse(accepts("{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{\"x\":[1,]}}]}", true));
        assertFalse(accepts("{\"tool_calls\":[{\"name\":\"read_file\",\"arguments\":{}}]", true));
    }

    private static boolean accepts(String text, boolean requiresCall) {
        var constraint = new JsonEnvelopeConstraint(tokenizer, List.of("read_file"), requiresCall);
        for (int id : tokenizer.encodeText(text)) {
            if (!constraint.allows(id)) return false;
            constraint.accept(id);
        }
        return constraint.complete();
    }

    private static void accept(JsonEnvelopeConstraint constraint, String text) {
        for (int id : tokenizer.encodeText(text)) {
            assertTrue(constraint.allows(id), "token " + id + " rejected after a valid JSON prefix");
            constraint.accept(id);
        }
    }
}
