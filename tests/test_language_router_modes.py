from src.router.language_router import LanguageMode, LanguageRouter


def test_ambiguous_whisper_metadata_does_not_become_english():
    result = LanguageRouter().route("lang maliku", whisper_lang="tl", whisper_prob=0.2968)

    assert result.mode is LanguageMode.MINGLISH


def test_whisper_urdu_metadata_is_preserved_without_markers():
    result = LanguageRouter().route("some roman text", whisper_lang="ur", whisper_prob=0.8)

    assert result.mode is LanguageMode.URDU


def test_english_and_minglish_examples_remain_stable():
    router = LanguageRouter()

    assert router.route("Hello, how are you?", whisper_lang="en", whisper_prob=0.9).mode is LanguageMode.ENGLISH
    assert router.route("Yaar, mujhe Python ka issue samjha do").mode is LanguageMode.MINGLISH
