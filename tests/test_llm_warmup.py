import urllib.error

from src.llm.llm_engine import LLMEngine


def test_model_loading_is_detected(monkeypatch):
    engine = LLMEngine()

    def fake_get_json(url, timeout=10):
        assert url.endswith("/api/ps")
        return {
            "models": [
                {"name": "deepseek-r1:1.5b", "status": "loading"},
            ]
        }

    monkeypatch.setattr("src.llm.llm_engine._get_json", fake_get_json)

    assert engine.is_model_loading() is True


def test_warmup_retries_until_model_loaded(monkeypatch):
    engine = LLMEngine()
    calls = {"chat": 0, "timeout": None}

    def fake_post_json(url, payload, timeout=20):
        calls["chat"] += 1
        calls["timeout"] = timeout
        return b"{}"

    monkeypatch.setattr("src.llm.llm_engine._post_json", fake_post_json)

    engine.warmup(timeout_seconds=180.0)

    assert calls == {"chat": 1, "timeout": 150}
    assert engine.num_ctx == 2048
    engine.warmup(timeout_seconds=180.0)
    assert calls["chat"] == 1


def test_warmup_timeout_is_not_normal_generation_timeout(monkeypatch):
    engine = LLMEngine()
    observed = {}

    def fake_post_json(url, payload, timeout=20):
        observed["timeout"] = timeout
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("src.llm.llm_engine._post_json", fake_post_json)

    try:
        engine.warmup(timeout_seconds=180.0)
    except TimeoutError:
        pass

    assert observed["timeout"] == 150
    assert engine.timeout == 20
