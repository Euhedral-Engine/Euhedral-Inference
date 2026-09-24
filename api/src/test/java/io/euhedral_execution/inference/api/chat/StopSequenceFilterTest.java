package io.euhedral_execution.inference.api.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class StopSequenceFilterTest {

    @Test
    void passesTextThroughWithoutStops() {
        var filter = new StopSequenceFilter(List.of());
        assertEquals("abc", filter.accept("abc"));
        assertEquals("", filter.finish());
        assertFalse(filter.matched());
    }

    @Test
    void holdsBackAPossibleStopPrefixUntilItResolves() {
        var filter = new StopSequenceFilter(List.of("END"));
        assertEquals("hello ", filter.accept("hello E"));
        assertEquals("", filter.accept("N"));
        assertEquals("ENx", filter.accept("x"), "a broken prefix is released");
        assertFalse(filter.matched());
    }

    @Test
    void matchSpanningChunksTruncatesAtTheStop() {
        var filter = new StopSequenceFilter(List.of("END"));
        assertEquals("a ", filter.accept("a E"));
        assertEquals("", filter.accept("ND tail"));
        assertTrue(filter.matched());
        assertEquals("", filter.accept("more"), "nothing is emitted after a match");
        assertEquals("", filter.finish());
    }

    @Test
    void earliestOfSeveralStopsWins() {
        var filter = new StopSequenceFilter(List.of("zz", "\n\n"));
        assertEquals("one", filter.accept("one\n\ntwo zz"));
        assertTrue(filter.matched());
    }

    @Test
    void finishReleasesHeldText() {
        var filter = new StopSequenceFilter(List.of("STOP"));
        assertEquals("x", filter.accept("xST"));
        assertEquals("ST", filter.finish());
    }
}
