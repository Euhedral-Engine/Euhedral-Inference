package io.euhedral_execution.inference.benchmark.result;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.euhedral_execution.inference.benchmark.measure.Metrics;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/// Reads and writes result rows as JSONL (one row per line, flushed as produced) or as one JSON
/// document `{"schema": "euhedral-inference.benchmark-results", "schemaVersion": 1, "results": [...]}`.
public final class ResultStore {
    public static final String DOCUMENT_SCHEMA = "euhedral-inference.benchmark-results";
    private static final Set<String> STATUSES =
            Set.of(BenchmarkResult.SUCCESS, BenchmarkResult.INELIGIBLE, BenchmarkResult.FAILED);

    private ResultStore() {}

    /// Rejects an existing output unless overwrite or append was requested.
    public static void checkOutput(Path output, boolean overwrite, boolean append) {
        if (Files.isDirectory(output)) throw new IllegalArgumentException("output is a directory: " + output);
        if (Files.exists(output) && !overwrite && !append)
            throw new IllegalArgumentException("output exists; pass --overwrite or --append: " + output);
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null && Files.exists(parent) && !Files.isWritable(parent))
            throw new IllegalArgumentException("output directory is not writable: " + parent);
    }

    /// Row sink. JSONL rows reach disk as they are produced; a JSON document is written on close.
    public static final class Writer implements AutoCloseable {
        private final Path output;
        private final boolean json;
        private final BufferedWriter lines;
        private final List<BenchmarkResult> rows = new ArrayList<>();

        public Writer(Path output, boolean json, boolean overwrite, boolean append) throws IOException {
            checkOutput(output, overwrite, append);
            Path parent = output.toAbsolutePath().getParent();
            if (parent != null) Files.createDirectories(parent);
            this.output = output;
            this.json = json;
            this.lines = json
                    ? null
                    : Files.newBufferedWriter(
                            output,
                            StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE,
                            StandardOpenOption.WRITE,
                            append ? StandardOpenOption.APPEND : StandardOpenOption.TRUNCATE_EXISTING);
        }

        public void write(BenchmarkResult row) {
            this.rows.add(row);
            if (this.json) return;
            try {
                this.lines.write(row.toJson());
                this.lines.newLine();
                this.lines.flush();
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
        }

        public List<BenchmarkResult> rows() {
            return List.copyOf(this.rows);
        }

        @Override
        public void close() throws IOException {
            if (!this.json) {
                this.lines.close();
                return;
            }
            ObjectNode document = BenchmarkResult.JSON.createObjectNode();
            document.put("schema", DOCUMENT_SCHEMA);
            document.put("schemaVersion", BenchmarkResult.SCHEMA_VERSION);
            document.set("results", BenchmarkResult.JSON.valueToTree(this.rows));
            Files.writeString(
                    this.output,
                    BenchmarkResult.JSON.writerWithDefaultPrettyPrinter().writeValueAsString(document) + "\n");
        }
    }

    /// Reads a JSON document, a JSON array of rows, or JSONL, validating every row.
    public static List<BenchmarkResult> read(Path input) throws IOException {
        String text = Files.readString(input);
        String trimmed = text.strip();
        List<JsonNode> nodes = new ArrayList<>();
        if (trimmed.startsWith("[")) {
            BenchmarkResult.JSON.readTree(trimmed).forEach(nodes::add);
        } else if (trimmed.startsWith("{") && isDocument(trimmed)) {
            JsonNode document = BenchmarkResult.JSON.readTree(trimmed);
            if (document.path("schemaVersion").asInt(-1) != BenchmarkResult.SCHEMA_VERSION)
                throw new IllegalArgumentException("unsupported document schemaVersion");
            document.path("results").forEach(nodes::add);
        } else {
            for (String line : text.split("\\R")) if (!line.isBlank()) nodes.add(BenchmarkResult.JSON.readTree(line));
        }
        List<BenchmarkResult> rows = new ArrayList<>();
        for (JsonNode node : nodes) rows.add(validate(BenchmarkResult.JSON.treeToValue(node, BenchmarkResult.class)));
        if (rows.isEmpty()) throw new IllegalArgumentException("no result rows in " + input);
        return rows;
    }

    private static boolean isDocument(String text) {
        try {
            return DOCUMENT_SCHEMA.equals(
                    BenchmarkResult.JSON.readTree(text).path("schema").asText());
        } catch (IOException notOneDocument) {
            return false;
        }
    }

    /// Enforces the schema contract that cannot be expressed by field types alone.
    static BenchmarkResult validate(BenchmarkResult row) {
        if (!BenchmarkResult.SCHEMA.equals(row.schema()))
            throw new IllegalArgumentException("unsupported row schema: " + row.schema());
        if (row.schemaVersion() != BenchmarkResult.SCHEMA_VERSION)
            throw new IllegalArgumentException("unsupported row schemaVersion: " + row.schemaVersion());
        if (row.implementation().isBlank()) throw new IllegalArgumentException("implementation must not be blank");
        if (!STATUSES.contains(row.status())) throw new IllegalArgumentException("unknown status: " + row.status());
        if (row.iteration() < 0) throw new IllegalArgumentException("iteration must not be negative");
        if (BenchmarkResult.FAILED.equals(row.status()) && row.throughput() != null)
            throw new IllegalArgumentException("failed rows must not report throughput");
        if (!BenchmarkResult.SUCCESS.equals(row.status())
                && (row.statusReason() == null || row.statusReason().isBlank()))
            throw new IllegalArgumentException("non-success rows need a statusReason");
        var work = row.work();
        for (Integer value : new Integer[] {
            work.promptTokens(),
            work.prefillQuanta(),
            work.prefillTokens(),
            work.generatedTokens(),
            work.decodeSampledTokens(),
            work.finalCommitQuanta()
        }) {
            if (value != null && value < 0) throw new IllegalArgumentException("work counts must not be negative");
        }
        var timings = row.timings();
        if (timings != null)
            for (Long value : new Long[] {
                timings.tokenization(),
                timings.prefill(),
                timings.firstTokenSample(),
                timings.timeToFirstToken(),
                timings.decode(),
                timings.decodeQuantaSum(),
                timings.finalCommit(),
                timings.timeToLastToken(),
                timings.endToEnd()
            }) {
                if (value != null && value < 0) throw new IllegalArgumentException("timings must not be negative");
            }
        var throughput = row.throughput();
        if (throughput != null)
            for (Double value : new Double[] {
                throughput.prefillTokensPerSecond(),
                throughput.decodeTokensPerSecond(),
                throughput.endToEndOutputTokensPerSecond()
            }) {
                if (value != null && (!Double.isFinite(value) || value <= 0))
                    throw new IllegalArgumentException("throughput must be finite and positive");
            }
        if (throughput != null && timings != null) {
            requireRate(
                    "prefillTokensPerSecond",
                    throughput.prefillTokensPerSecond(),
                    Metrics.perSecond(orZero(work.prefillTokens()), timings.prefill()));
            requireRate(
                    "decodeTokensPerSecond",
                    throughput.decodeTokensPerSecond(),
                    Metrics.perSecond(orZero(work.decodeSampledTokens()), timings.decode()));
            requireRate(
                    "endToEndOutputTokensPerSecond",
                    throughput.endToEndOutputTokensPerSecond(),
                    Metrics.perSecond(orZero(work.generatedTokens()), timings.endToEnd()));
        }
        return row;
    }

    private static int orZero(Integer value) {
        return value == null ? 0 : value;
    }

    /// A reported rate must equal this schema's definition over the row's own work and timing.
    private static void requireRate(String name, Double reported, Double expected) {
        if (reported == null && expected == null) return;
        if (reported == null || expected == null || Math.abs(reported - expected) > 1e-9 * Math.max(1.0, expected))
            throw new IllegalArgumentException(
                    name + " " + reported + " does not match its work and timing (" + expected + ")");
    }

    /// Relabels external rows as imported. Each row must already name `implementation`, so an
    /// import cannot silently turn one implementation's result into another's.
    public static List<BenchmarkResult> importRows(Path input, String implementation, String note) throws IOException {
        Objects.requireNonNull(implementation, "implementation");
        if (implementation.isBlank()) throw new IllegalArgumentException("--implementation is required");
        String sha256 = java.util.HexFormat.of().formatHex(sha256(Files.readAllBytes(input)));
        List<BenchmarkResult> imported = new ArrayList<>();
        for (BenchmarkResult row : read(input)) {
            if (!implementation.equals(row.implementation()))
                throw new IllegalArgumentException("row implementation " + row.implementation()
                        + " does not match --implementation " + implementation);
            var original = row.provenance();
            imported.add(row.withSource(
                    implementation,
                    new BenchmarkResult.Provenance(
                            "imported",
                            original.tool(),
                            original.commandLine(),
                            input.toAbsolutePath().normalize().toString(),
                            sha256,
                            note)));
        }
        return imported;
    }

    private static byte[] sha256(byte[] bytes) {
        try {
            return java.security.MessageDigest.getInstance("SHA-256").digest(bytes);
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }
}
