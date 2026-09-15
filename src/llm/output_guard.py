"""
ARIA V3 - Incremental Streaming Output Guard
Implements:
1. Streaming State Machine for <think>...</think> suppression (0% leakage across chunk splits)
2. Markdown syntax stripping (*, #, `, _, bullet points)
3. Deterministic script validation: catches Arabic/Urdu Unicode ([\u0600-\u06FF]) and transliterates or flags retry
4. Strict response length clamping: max 400 characters, max 3 sentences
5. Processing overhead < 1 ms per token chunk
"""

import enum
import re
import time
from typing import Generator, List, Optional, Tuple


class GuardState(enum.Enum):
    OUTSIDE_THINK = 0
    INSIDE_THINK = 1


# Arabic/Urdu Unicode range
ARABIC_SCRIPT_RE = re.compile(r'[\u0600-\u06FF]')

# CJK Unicode range
CJK_SCRIPT_RE = re.compile(r'[\u4E00-\u9FFF]')

# Markdown stripping patterns
MARKDOWN_RE = re.compile(r'[*_~`#>]|^[\s]*[-*+]\s+|^[\s]*\d+\.\s+', re.MULTILINE)
CODE_BLOCK_RE = re.compile(r'```.*?```', re.DOTALL)
INLINE_CODE_RE = re.compile(r'`[^`]+`')
URL_RE = re.compile(r'https?://\S+|www\.\S+')

# Sentence delimiter regex (punctuation followed by space/end or newline)
SENTENCE_SPLIT_RE = re.compile(r'([^.!?\n]+[.!?\n]*)', re.UNICODE)

# Common deterministic phonetic map for basic Urdu Nastaliq words
PHONETIC_URDU_MAP = {
    "سلام": "Salam",
    "وعلیکم": "Walaikum",
    "شکریہ": "Shukriya",
    "ہاں": "Haan",
    "نہیں": "Nahi",
    "ٹھیک": "Theek",
    "کیا": "Kya",
    "ہے": "Hai",
    "ہیں": "Hain",
    "آپ": "Aap",
    "کیسے": "Kaisay",
    "کون": "Kaun",
    "میرا": "Mera",
    "نام": "Naam",
    "معاف": "Maaf",
    "کیجئے": "Kijiye",
    "اللہ": "Allah",
    "حافظ": "Hafiz",
}


def strip_markdown(text: str, preserve_leading_space: bool = False) -> str:
    """Remove markdown headers, bold, italics, bullets, inline code.

    Args:
        preserve_leading_space: If True, a single leading space is kept.
            Use for streaming chunks where the space is a word boundary.
    """
    leading_space = text.startswith(' ') and preserve_leading_space
    text = CODE_BLOCK_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MARKDOWN_RE.sub("", text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    if leading_space and text:
        text = ' ' + text
    return text


def sanitize_script(text: str) -> Tuple[str, bool, bool]:
    """
    Check for forbidden Unicode scripts.
    Applies phonetic mapping for recognized words where possible.
    Returns: (cleaned_text, had_arabic, is_corrupt)
    """
    had_arabic = bool(ARABIC_SCRIPT_RE.search(text))
    had_cjk = bool(CJK_SCRIPT_RE.search(text))

    if not had_arabic and not had_cjk:
        return text, False, False

    out_tokens = []
    is_corrupt = False

    for token in text.split(" "):
        if ARABIC_SCRIPT_RE.search(token):
            # Clean punctuation from token for lookup
            clean_tok = re.sub(r'[^\u0600-\u06FF]', '', token)
            if clean_tok in PHONETIC_URDU_MAP:
                out_tokens.append(PHONETIC_URDU_MAP[clean_tok])
            else:
                # Unmapped Arabic script
                is_corrupt = True
        elif CJK_SCRIPT_RE.search(token):
            is_corrupt = True
        else:
            out_tokens.append(token)

    sanitized = " ".join(out_tokens).strip()
    return sanitized, had_arabic or had_cjk, is_corrupt


def count_sentences(text: str) -> List[str]:
    """Extract individual sentences using punctuation boundaries."""
    raw_sentences = SENTENCE_SPLIT_RE.findall(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


class OutputGuard:
    """
    Incremental streaming guard for ARIA voice pipeline.
    Maintains internal state across chunks to guarantee:
    - 0% reasoning leakage across token boundaries
    - Markdown removal
    - Script safety (Arabic Unicode filter)
    - Response length clamping (<= 400 chars, <= 3 sentences)
    """

    def __init__(
        self,
        max_chars: int = 400,
        max_sentences: int = 3,
        allow_retries: bool = True
    ):
        self.max_chars = max_chars
        self.max_sentences = max_sentences
        self.allow_retries = allow_retries

        self.reset()

    def reset(self, max_chars: Optional[int] = None, max_sentences: Optional[int] = None):
        """Reset state for a new generation turn."""
        if max_chars is not None:
            self.max_chars = max_chars
        if max_sentences is not None:
            self.max_sentences = max_sentences
        self.state = GuardState.OUTSIDE_THINK
        self.raw_buffer = ""
        self.think_buffer = ""
        self.content_buffer = ""
        self.emitted_chars = 0
        self.emitted_sentences = 0
        self.is_clamped = False
        self.script_violation = False
        self.requires_retry = False
        self.total_tokens_processed = 0
        self.total_processing_time_s = 0.0

    def process_chunk(self, chunk: str) -> str:
        """
        Process an incoming raw streaming token/chunk.
        Returns the sanitized text ready for downstream sentence buffering,
        or empty string if inside think block or clamped.
        """
        if self.is_clamped:
            return ""

        t0 = time.perf_counter()
        self.total_tokens_processed += 1

        self.raw_buffer += chunk
        output_tokens = []

        while self.raw_buffer:
            if self.state == GuardState.OUTSIDE_THINK:
                open_pos = self.raw_buffer.lower().find("<think>")
                if open_pos != -1:
                    # Content before <think>
                    pre_content = self.raw_buffer[:open_pos]
                    if pre_content:
                        output_tokens.append(pre_content)
                    self.raw_buffer = self.raw_buffer[open_pos + 7:]
                    self.state = GuardState.INSIDE_THINK
                else:
                    # Check for partial open tag at end of buffer
                    partial_idx = self._check_partial_tag(self.raw_buffer, "<think>")
                    if partial_idx is not None:
                        safe_text = self.raw_buffer[:partial_idx]
                        if safe_text:
                            output_tokens.append(safe_text)
                        self.raw_buffer = self.raw_buffer[partial_idx:]
                        break
                    else:
                        output_tokens.append(self.raw_buffer)
                        self.raw_buffer = ""
                        break

            elif self.state == GuardState.INSIDE_THINK:
                close_pos = self.raw_buffer.lower().find("</think>")
                if close_pos != -1:
                    think_text = self.raw_buffer[:close_pos]
                    self.think_buffer += think_text
                    self.raw_buffer = self.raw_buffer[close_pos + 8:]
                    self.state = GuardState.OUTSIDE_THINK
                else:
                    # Check for partial close tag at end of buffer
                    partial_idx = self._check_partial_tag(self.raw_buffer, "</think>")
                    if partial_idx is not None:
                        think_text = self.raw_buffer[:partial_idx]
                        self.think_buffer += think_text
                        self.raw_buffer = self.raw_buffer[partial_idx:]
                        break
                    else:
                        self.think_buffer += self.raw_buffer
                        self.raw_buffer = ""
                        break

        raw_emit = "".join(output_tokens)
        if not raw_emit:
            self.total_processing_time_s += (time.perf_counter() - t0)
            return ""

        # Post-process emitted content: markdown & script filtering
        # preserve_leading_space=True so streaming word boundaries (" word") are kept
        clean_emit = strip_markdown(raw_emit, preserve_leading_space=True)
        clean_emit, had_arabic, is_corrupt = sanitize_script(clean_emit)

        if had_arabic or is_corrupt:
            self.script_violation = True
            if is_corrupt:
                self.requires_retry = True

        # Length clamping checks
        to_emit = ""
        for ch in clean_emit:
            if self.emitted_chars >= self.max_chars:
                self.is_clamped = True
                break

            to_emit += ch
            self.emitted_chars += 1

            if ch in ".!?\n":
                # Check sentence count
                current_full = self.content_buffer + to_emit
                sents = count_sentences(current_full)
                if len(sents) >= self.max_sentences:
                    self.is_clamped = True
                    break

        self.content_buffer += to_emit
        self.total_processing_time_s += (time.perf_counter() - t0)
        return to_emit

    def flush(self) -> str:
        """
        Flush any remaining buffer at end of stream.
        """
        if self.is_clamped or self.state == GuardState.INSIDE_THINK:
            self.raw_buffer = ""
            return ""

        clean = strip_markdown(self.raw_buffer)
        clean, had_arabic, is_corrupt = sanitize_script(clean)
        if had_arabic or is_corrupt:
            self.script_violation = True
            if is_corrupt:
                self.requires_retry = True

        remaining = clean[:max(0, self.max_chars - self.emitted_chars)]
        self.content_buffer += remaining
        self.raw_buffer = ""
        return remaining

    def get_full_response(self) -> str:
        """Return full accumulated user-facing response."""
        return self.content_buffer.strip()

    def get_metrics(self) -> dict:
        """Return benchmark and diagnostic metrics."""
        avg_overhead_us = (
            (self.total_processing_time_s / self.total_tokens_processed) * 1_000_000
            if self.total_tokens_processed > 0 else 0.0
        )
        return {
            "total_tokens_processed": self.total_tokens_processed,
            "total_processing_time_ms": round(self.total_processing_time_s * 1000, 3),
            "avg_overhead_per_chunk_us": round(avg_overhead_us, 2),
            "avg_overhead_per_chunk_ms": round(avg_overhead_us / 1000, 4),
            "think_chars_suppressed": len(self.think_buffer),
            "content_chars_emitted": len(self.content_buffer),
            "content_sentences": len(count_sentences(self.content_buffer)),
            "is_clamped": self.is_clamped,
            "script_violation": self.script_violation,
            "requires_retry": self.requires_retry,
        }

    @staticmethod
    def _check_partial_tag(text: str, tag: str) -> Optional[int]:
        """
        Detect if text ends with a prefix of tag (e.g. text ends with '<th').
        Returns index in text where the partial tag begins, or None.
        """
        text_lower = text.lower()
        tag_lower = tag.lower()
        for i in range(1, len(tag_lower)):
            prefix = tag_lower[:i]
            if text_lower.endswith(prefix):
                return len(text) - len(prefix)
        return None
