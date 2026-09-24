package io.euhedral_execution.inference.core.tokenizer;

import java.util.Arrays;
import java.util.Objects;

/// Converts generated token IDs to text while retaining an incomplete UTF-8 suffix between calls.
public final class IncrementalDecoder {
    private final QwenTokenizer tokenizer;
    private byte[] pendingBytes = new byte[32];
    private int pendingLength;
    private boolean finished;

    public IncrementalDecoder(QwenTokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    /// Appends one token and returns only complete UTF-8 text safe to emit to a caller.
    public String append(int tokenId) {
        ensureOpen();
        this.tokenizer.requireTokenId(tokenId);
        if (this.tokenizer.isControlTokenId(tokenId)) {
            String prefix = flushPendingBytes();
            return prefix + this.tokenizer.tokenText(tokenId);
        }
        String token = this.tokenizer.tokenText(tokenId);
        for (int index = 0; index < token.length(); ) {
            int codePoint = token.codePointAt(index);
            int value = this.tokenizer.byteForCodePoint(codePoint);
            if (value < 0) throw new IllegalArgumentException("Token is not encoded with Qwen byte-level BPE");
            appendByte((byte) value);
            index += Character.charCount(codePoint);
        }
        return emitCompleteUtf8();
    }

    /// Appends a token chunk and returns the newly completed text.
    public String append(int[] tokenIds) {
        ensureOpen();
        Objects.requireNonNull(tokenIds, "tokenIds");
        StringBuilder output = new StringBuilder();
        for (int tokenId : tokenIds) output.append(append(tokenId));
        return output.toString();
    }

    /// Completes decoding; an invalid truncated suffix is represented by the UTF-8 replacement character.
    public String finish() {
        ensureOpen();
        this.finished = true;
        if (this.pendingLength == 0) return "";
        String remaining =
                new String(this.pendingBytes, 0, this.pendingLength, java.nio.charset.StandardCharsets.UTF_8);
        this.pendingLength = 0;
        return remaining;
    }

    private void appendByte(byte value) {
        if (this.pendingLength == this.pendingBytes.length) {
            this.pendingBytes = Arrays.copyOf(this.pendingBytes, this.pendingBytes.length * 2);
        }
        this.pendingBytes[this.pendingLength++] = value;
    }

    private String emitCompleteUtf8() {
        int safeLength = completeUtf8Length(this.pendingBytes, this.pendingLength);
        if (safeLength == 0) return "";
        String output = new String(this.pendingBytes, 0, safeLength, java.nio.charset.StandardCharsets.UTF_8);
        this.pendingLength -= safeLength;
        System.arraycopy(this.pendingBytes, safeLength, this.pendingBytes, 0, this.pendingLength);
        return output;
    }

    private String flushPendingBytes() {
        if (this.pendingLength == 0) return "";
        String output = new String(this.pendingBytes, 0, this.pendingLength, java.nio.charset.StandardCharsets.UTF_8);
        this.pendingLength = 0;
        return output;
    }

    private void ensureOpen() {
        if (this.finished) throw new IllegalStateException("incremental Qwen decoder is already finished");
    }

    private static int completeUtf8Length(byte[] bytes, int length) {
        int offset = 0;
        while (offset < length) {
            int first = bytes[offset] & 0xff;
            int expected = first <= 0x7f
                    ? 1
                    : first >= 0xc2 && first <= 0xdf
                            ? 2
                            : first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 1;
            if (expected == 1) {
                offset++;
                continue;
            }
            int available = length - offset;
            int validContinuationCount = 0;
            while (validContinuationCount < expected - 1 && validContinuationCount + 1 < available) {
                int next = bytes[offset + validContinuationCount + 1] & 0xff;
                if (next < 0x80 || next > 0xbf) break;
                validContinuationCount++;
            }
            if (validContinuationCount < expected - 1) {
                if (validContinuationCount == available - 1) return offset;
                offset++;
                continue;
            }
            offset += expected;
        }
        return offset;
    }
}
