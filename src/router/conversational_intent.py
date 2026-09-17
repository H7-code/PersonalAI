"""Small deterministic conversational shortcuts for common voice greetings."""

from dataclasses import dataclass
import re
from typing import Optional

from src.router.language_router import LanguageMode


@dataclass(frozen=True)
class IntentResponse:
    intent: str
    mode: LanguageMode
    response: str


def normalize_known_variants(text: str) -> str:
    """Normalize spelling variants used by the small greeting matcher only."""
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    normalized = re.sub(r"\b(?:assalamualaikum|asalamualaikum)\b", "assalam o alaikum", normalized)
    normalized = re.sub(r"\bassalam\b|\basalam\b", "assalam", normalized)
    normalized = re.sub(r"\bkaisay\b", "kaise", normalized)
    normalized = re.sub(r"\bhay\b", "hain", normalized)
    normalized = re.sub(r"\bhoo\b", "ho", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def detect_conversational_intent(text: str) -> Optional[IntentResponse]:
    """Return a short response for known greetings, or None for normal LLM routing."""
    value = normalize_known_variants(text)

    if (
        re.search(r"\bassalam\s+(?:o\s+)?alaikum\b", value)
        and re.search(r"\bhow\s+are\s+you\b", value)
    ):
        return IntentResponse(
            "minglish_greeting",
            LanguageMode.URDU,
            "Wa alaikum assalam! Main theek hoon. How are you?",
        )
    if "yaar kya haal hai" in value:
        return IntentResponse(
            "minglish_wellbeing",
            LanguageMode.URDU,
            "Main theek hoon yaar. Aap batao?",
        )
    if re.search(r"\bhello\s+aria\b", value) and re.search(r"\bkya\s+kar\s+rahi\s+ho\b", value):
        return IntentResponse(
            "minglish_presence",
            LanguageMode.URDU,
            "Hello! Main yahan hoon aur aap ki help ke liye ready hoon.",
        )

    if re.search(r"\bassalam\s+(?:o\s+)?alaikum\b", value) or value == "salam":
        return IntentResponse(
            "urdu_greeting",
            LanguageMode.URDU,
            "Wa alaikum assalam. Aap kaise hain?",
        )
    if re.search(r"\bkya\s+haal\s+hai(?:n)?\b", value):
        return IntentResponse(
            "urdu_wellbeing",
            LanguageMode.URDU,
            "Alhamdulillah, main theek hoon. Aap kaise hain?",
        )
    if re.search(r"\baap\s+kaise\s+(?:ho|hain)\b", value):
        return IntentResponse(
            "urdu_wellbeing",
            LanguageMode.URDU,
            "Main theek hoon. Aap kaise hain?",
        )

    if value == "hello":
        return IntentResponse("english_greeting", LanguageMode.ENGLISH, "Hello! How are you?")
    if value == "hi":
        return IntentResponse("english_greeting", LanguageMode.ENGLISH, "Hi! How are you?")
    if value == "good morning":
        return IntentResponse("english_greeting", LanguageMode.ENGLISH, "Good morning! How are you today?")
    if value == "how are you":
        return IntentResponse("english_wellbeing", LanguageMode.ENGLISH, "I'm doing well. How are you?")
    if re.fullmatch(r"hey(?: aria)?", value):
        return IntentResponse("english_greeting", LanguageMode.ENGLISH, "Hey! How can I help you?")

    return None
