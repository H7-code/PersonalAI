"""
ARIA Deterministic Language Router
Classifies incoming transcripts into ENGLISH or URDU. English/Urdu mixed speech uses URDU mode.
Execution latency strictly < 2 ms.
"""

import enum
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple

from src.stt.transliteration import transliterate_to_roman_urdu, is_perso_arabic

logger = logging.getLogger("aria.router")

class LanguageMode(str, enum.Enum):
    ENGLISH = "ENGLISH"
    URDU = "URDU"

@dataclass
class RoutingResult:
    mode: LanguageMode
    confidence: float
    latency_ms: float
    english_ratio: float
    urdu_ratio: float
    urdu_tokens_count: int
    english_tokens_count: int
    total_tokens_count: int
    matched_urdu_markers: List[str]
    matched_english_words: List[str]
    is_ambiguous_boundary: bool = False
    fallback_reason: Optional[str] = None


class LanguageRouter:
    """
    Deterministic Vocabulary and Script Heuristic Classifier.
    Combines token dictionary analysis, script normalization, and Whisper metadata.
    """

    # Curated Roman Urdu Marker Lexicon (Function words, pronouns, auxiliaries, common verbs)
    # Homographs shared with English ('the', 'in', 'me', 'is', 'to', 'so', 'no', 'us') are strictly excluded
    ROMAN_URDU_MARKERS: Set[str] = {
        "kya", "kyun", "kab", "kahan", "kaise", "kaisay", "kaisa", "kaisi", "kaun", "koun",
        "hai", "hain", "ho", "hoon", "tha", "thi", "thay", "hoga", "hogi", "hoge",
        "aap", "aapka", "aapki", "aapke", "tum", "tumhara", "main", "mera", "meri", "mere",
        "hum", "hamara", "hamari", "hamare", "yeh", "ye", "woh", "wo", "iss", "uss", "un", "inn",
        "ka", "ki", "ke", "ko", "se", "mein", "par", "tak", "saath", "liye", "ne",
        "nahi", "nahin", "na", "mat", "haan", "theek", "shukriya", "bilkul", "acha", "achi", "ache",
        "bohot", "bahut", "zyada", "thora", "thori", "thore", "kam", "kuch", "sab", "har",
        "kar", "karo", "karna", "karte", "karti", "karta", "kiya", "karein", "kare",
        "raha", "rahi", "rahe", "rahey", "gaya", "gayi", "gaye",
        "soch", "sochna", "dekh", "dekho", "dekhna", "batao", "batana", "bolo", "bolna", "suno", "sunna",
        "madad", "sawaal", "sawal", "jawab", "baat", "naam", "din", "waqt", "pehle", "phir", "baad",
        "chahta", "chahti", "chahte", "sakta", "sakti", "sakte", "chahiye", "yaqeen", "zaroor",
        "khana", "kha", "liya", "diya", "dekha", "suna", "pareshan", "subah", "shaam", "raat",
        "kitaab", "ghar", "kaam", "dhyan", "chalo", "jao",
    }

    # Common English Lexicon (Function words, common nouns, system terminology)
    ENGLISH_LEXICON: Set[str] = {
        "the", "a", "an", "is", "are", "am", "was", "were", "be", "been", "being",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about", "into", "over", "after",
        "and", "but", "or", "so", "if", "because", "while", "though", "since",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "their", "our", "its",
        "this", "that", "these", "those", "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must", "do", "does", "did", "done",
        "have", "has", "had", "go", "going", "gone", "make", "get", "take", "see", "look", "know", "think", "come",
        "give", "tell", "say", "said", "ask", "help", "hello", "hi", "hey", "please", "thanks", "thank", "okay",
        "yes", "no", "not", "all", "some", "any", "good", "great", "day", "today", "now", "here", "there",
        "system", "cpu", "gpu", "ram", "memory", "usage", "network", "connection", "download", "upload",
        "update", "model", "load", "file", "disk", "space", "error", "message", "status", "restart", "start",
        "stop", "crash", "app", "application", "server", "service", "settings", "privacy", "disable", "enable",
        "telemetry", "parameters", "normal", "processing", "moment", "percent", "megabytes", "dependency",
        "transformer", "neural", "pytorch", "tensorflow", "cuda", "api", "json", "http",
        "backup", "delete", "restore", "port", "summarize", "performance", "latest", "database",
        "bye", "goodbye",
    }

    # Short Ambiguous Boundary Tokens
    BOUNDARY_SHORT_WORDS: Dict[str, LanguageMode] = {
        "ok": LanguageMode.ENGLISH,
        "okay": LanguageMode.ENGLISH,
        "done": LanguageMode.ENGLISH,
        "yes": LanguageMode.ENGLISH,
        "no": LanguageMode.ENGLISH,
        "fine": LanguageMode.ENGLISH,
        "sure": LanguageMode.ENGLISH,
        "right": LanguageMode.ENGLISH,
        "hello": LanguageMode.ENGLISH,
        "thanks": LanguageMode.ENGLISH,
        "haan": LanguageMode.URDU,
        "nahin": LanguageMode.URDU,
        "nahi": LanguageMode.URDU,
        "shukriya": LanguageMode.URDU,
        "theek": LanguageMode.URDU,
        "jee": LanguageMode.URDU,
        "ji": LanguageMode.URDU,
    }

    def __init__(
        self,
        urdu_ratio_threshold: float = 0.55,
        english_ratio_threshold: float = 0.75,
    ):
        self.urdu_ratio_threshold = urdu_ratio_threshold
        self.english_ratio_threshold = english_ratio_threshold

    def route(
        self,
        text: str,
        whisper_lang: Optional[str] = None,
        whisper_prob: Optional[float] = None,
    ) -> RoutingResult:
        """
        Routes transcript text into ENGLISH, URDU, or MINGLISH.
        Guarantees sub-millisecond execution latency (< 2 ms).
        """
        t0 = time.perf_counter()

        # Step 1: Character Set & Script Check
        if is_perso_arabic(text):
            text = transliterate_to_roman_urdu(text)

        # Step 2: Tokenization & Dictionary Lookup
        clean_text = text.lower()
        tokens = re.findall(r"[a-z0-9\-_']+", clean_text)
        total_tokens = len(tokens)

        has_salam_greeting = bool(re.search(r"\bass?alam(?:ualaikum|\s+(?:o\s+)?alaikum)\b|\bsalam\b", clean_text))
        has_english_clause = bool(re.search(r"\b(?:how\s+are\s+you|hello|hi|hey)\b", clean_text))
        if has_salam_greeting:
            mode = LanguageMode.URDU
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return RoutingResult(
                mode=mode,
                confidence=0.98,
                latency_ms=round(dur_ms, 3),
                english_ratio=0.0,
                urdu_ratio=1.0,
                urdu_tokens_count=total_tokens,
                english_tokens_count=0,
                total_tokens_count=total_tokens,
                matched_urdu_markers=["greeting"],
                matched_english_words=[],
                is_ambiguous_boundary=False,
                fallback_reason=None,
            )

        if total_tokens == 0:
            # Empty input fallback
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return RoutingResult(
                mode=LanguageMode.ENGLISH,
                confidence=1.0,
                latency_ms=dur_ms,
                english_ratio=0.0,
                urdu_ratio=0.0,
                urdu_tokens_count=0,
                english_tokens_count=0,
                total_tokens_count=0,
                matched_urdu_markers=[],
                matched_english_words=[],
                is_ambiguous_boundary=True,
                fallback_reason="Empty transcript",
            )

        # Special boundary handling for 1-word or 2-word short utterances
        if total_tokens <= 2:
            first_token = tokens[0]
            if total_tokens == 1 and first_token in self.BOUNDARY_SHORT_WORDS:
                dur_ms = (time.perf_counter() - t0) * 1000.0
                mode = self.BOUNDARY_SHORT_WORDS[first_token]
                return RoutingResult(
                    mode=mode,
                    confidence=0.95,
                    latency_ms=dur_ms,
                    english_ratio=1.0 if mode == LanguageMode.ENGLISH else 0.0,
                    urdu_ratio=1.0 if mode == LanguageMode.URDU else 0.0,
                    urdu_tokens_count=1 if mode == LanguageMode.URDU else 0,
                    english_tokens_count=1 if mode == LanguageMode.ENGLISH else 0,
                    total_tokens_count=1,
                    matched_urdu_markers=[first_token] if mode == LanguageMode.URDU else [],
                    matched_english_words=[first_token] if mode == LanguageMode.ENGLISH else [],
                    is_ambiguous_boundary=True,
                    fallback_reason="Short boundary utterance match",
                )

        matched_urdu = [t for t in tokens if t in self.ROMAN_URDU_MARKERS]
        matched_english = [t for t in tokens if t in self.ENGLISH_LEXICON]

        urdu_count = len(matched_urdu)
        english_count = len(matched_english)

        urdu_ratio = urdu_count / total_tokens
        english_ratio = english_count / total_tokens

        # Step 3: Calibrated Decision Logic
        # Case A: Strict English only when text or reliable Whisper metadata supports it.
        # Ambiguous non-English metadata must not silently become English.
        if urdu_count == 0 and (
            (whisper_lang == "en" and (whisper_prob is None or whisper_prob >= 0.60))
            or (english_count > 0 and (whisper_lang in {None, "en"}))
        ):
            mode = LanguageMode.ENGLISH
            confidence = max(0.85, english_ratio)
            fallback_reason = None

        elif urdu_count == 0 and whisper_lang == "ur":
            mode = LanguageMode.URDU
            confidence = whisper_prob if whisper_prob is not None else 0.70
            fallback_reason = "Whisper Urdu metadata"

        elif urdu_count == 0:
            mode = LanguageMode.URDU
            confidence = 0.55
            fallback_reason = "Ambiguous language metadata"

        # Case B: Strict Roman Urdu (High Urdu marker density, minimal/no English vocabulary)
        elif urdu_ratio >= self.urdu_ratio_threshold and english_count == 0:
            mode = LanguageMode.URDU
            confidence = min(0.98, urdu_ratio + 0.2)
            fallback_reason = None

        # Case C: English/Urdu mixed speech uses the Urdu response/TTS path.
        elif urdu_count > 0 and english_count > 0:
            mode = LanguageMode.URDU
            confidence = 0.90
            fallback_reason = None

        # Case D: Mixed / Ambiguous boundary cases
        elif urdu_count > 0 and english_count == 0:
            # Urdu markers present, but other tokens are uncatalogued (e.g. proper nouns or loan words)
            if urdu_ratio >= 0.50:
                mode = LanguageMode.URDU
                confidence = 0.80
            else:
                mode = LanguageMode.URDU
                confidence = 0.75
            fallback_reason = "Uncatalogued tokens with Urdu markers present"

        else:
            # Fallback
            if whisper_lang == "ur":
                mode = LanguageMode.URDU
            elif whisper_lang == "en":
                mode = LanguageMode.ENGLISH
            else:
                mode = LanguageMode.URDU
            confidence = 0.65
            fallback_reason = "Whisper language metadata fallback"

        dur_ms = (time.perf_counter() - t0) * 1000.0

        return RoutingResult(
            mode=mode,
            confidence=round(confidence, 2),
            latency_ms=round(dur_ms, 3),
            english_ratio=round(english_ratio, 4),
            urdu_ratio=round(urdu_ratio, 4),
            urdu_tokens_count=urdu_count,
            english_tokens_count=english_count,
            total_tokens_count=total_tokens,
            matched_urdu_markers=matched_urdu,
            matched_english_words=matched_english,
            is_ambiguous_boundary=(fallback_reason is not None),
            fallback_reason=fallback_reason,
        )
