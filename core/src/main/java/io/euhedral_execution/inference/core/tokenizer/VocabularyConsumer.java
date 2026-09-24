package io.euhedral_execution.inference.core.tokenizer;

import java.io.IOException;

@FunctionalInterface
interface VocabularyConsumer {
    void accept(String token, int id) throws IOException;
}
