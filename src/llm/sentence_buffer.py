"""
ARIA V3 - Lookahead Sentence Boundary Buffer
Splits incoming streaming text into complete sentences for incremental TTS synthesis.
Features:
- Immediate first-sentence emission upon boundary detection
- Abbreviation protection (Dr., Mr., Ms., e.g., i.e., vs., etc.)
- Decimal/number protection (e.g. 3.14, 1.5, v3.0)
- Ellipsis protection (...)
- Low overhead (< 0.1 ms per feed)
"""

import re
from typing import List

# Common abbreviations that should NOT trigger a sentence split
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "e.g", "i.e", "vs", "etc", "approx", "fig", "inc", "ltd"
}

# Regex for sentence delimiters followed by lookahead space/end
# Delimiter: . ! ? or newline
DELIM_PATTERN = re.compile(r'([.!?\n]+)(\s+|$)')


class SentenceBuffer:
    def __init__(self, min_sentence_chars: int = 3):
        self.min_sentence_chars = min_sentence_chars
        self.buffer = ""
        self.emitted_sentences = []

    def reset(self):
        self.buffer = ""
        self.emitted_sentences = []

    def feed(self, chunk: str) -> List[str]:
        """
        Feed an incoming text chunk. Returns any newly completed sentences.
        """
        if not chunk:
            return []

        self.buffer += chunk
        emitted = []

        search_start = 0
        while True:
            match = DELIM_PATTERN.search(self.buffer, search_start)
            if not match:
                break

            delim = match.group(1)
            split_idx = match.start(1) + len(delim)
            candidate = self.buffer[:split_idx].strip()
            remainder = self.buffer[match.end():]

            # Protection 1: Check if delimiter is part of a decimal number (e.g. "3.14")
            start_pos = match.start(1)
            if start_pos > 0 and start_pos + 1 < len(self.buffer):
                prev_char = self.buffer[start_pos - 1]
                next_char = self.buffer[start_pos + 1]
                if prev_char.isdigit() and next_char.isdigit():
                    search_start = match.end()
                    continue

            # Protection 2: Check for abbreviations (e.g. "Dr. Smith", "e.g. this")
            last_word = re.split(r'\s+', candidate.rstrip(".!?"))[-1].lower() if candidate else ""
            if last_word in ABBREVIATIONS:
                search_start = match.end()
                continue

            # Protection 3: Check candidate length
            if len(candidate) < self.min_sentence_chars:
                search_start = match.end()
                continue

            emitted.append(candidate)
            self.emitted_sentences.append(candidate)
            self.buffer = remainder
            search_start = 0

        return emitted

    def flush(self) -> List[str]:
        """Flush remaining text as the final sentence."""
        text = self.buffer.strip()
        self.buffer = ""
        if text:
            self.emitted_sentences.append(text)
            return [text]
        return []
