import pytest

from src.router.conversational_intent import detect_conversational_intent
from src.router.language_router import LanguageMode


@pytest.mark.parametrize(
    ("text", "mode", "response"),
    [
        ("Assalam o alaikum", LanguageMode.URDU, "Wa alaikum assalam. Aap kaise hain?"),
        ("Asalamualaikum", LanguageMode.URDU, "Wa alaikum assalam. Aap kaise hain?"),
        ("Kya haal hay?", LanguageMode.URDU, "Alhamdulillah, main theek hoon. Aap kaise hain?"),
        ("Hey ARIA", LanguageMode.ENGLISH, "Hey! How can I help you?"),
        ("How are you?", LanguageMode.ENGLISH, "I'm doing well. How are you?"),
        (
            "Assalam o alaikum, how are you?",
            LanguageMode.MINGLISH,
            "Wa alaikum assalam! Main theek hoon. How are you?",
        ),
    ],
)
def test_common_conversational_intents(text, mode, response):
    result = detect_conversational_intent(text)

    assert result is not None
    assert result.mode is mode
    assert result.response == response


def test_unknown_text_falls_through_to_llm():
    assert detect_conversational_intent("Explain Python decorators") is None
