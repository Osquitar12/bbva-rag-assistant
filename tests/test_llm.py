import httpx
import pytest

from app.config import Settings
from app.exceptions import LLMConfigurationError, LLMError
from app.rag.llm import GroqLLM, LLMFactory, OllamaLLM


def _ok(request: httpx.Request) -> httpx.Response:
    assert request.url.path.endswith("/chat/completions")
    return httpx.Response(200, json={"choices": [{"message": {"content": "  hola  "}}]})


def test_factory_creates_groq_and_generates():
    s = Settings(llm_provider="groq", groq_api_key="k")
    llm = LLMFactory.create(s, transport=httpx.MockTransport(_ok))
    assert isinstance(llm, GroqLLM)
    assert llm.generate([{"role": "user", "content": "hi"}]) == "hola"


def test_factory_creates_ollama():
    s = Settings(llm_provider="ollama")
    assert isinstance(LLMFactory.create(s, transport=httpx.MockTransport(_ok)), OllamaLLM)


def test_missing_api_key_and_unknown_provider():
    with pytest.raises(LLMConfigurationError):
        LLMFactory.create(Settings(llm_provider="groq", groq_api_key=""))
    with pytest.raises(LLMConfigurationError):
        LLMFactory.create(Settings(llm_provider="nope"))


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("app.rag.llm.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] < 2 else _ok(request)

    llm = LLMFactory.create(Settings(llm_provider="groq", groq_api_key="k"), transport=httpx.MockTransport(flaky))
    assert llm.generate([{"role": "user", "content": "x"}]) == "hola"
    assert calls["n"] == 2


def test_auth_error_and_exhausted_retries(monkeypatch):
    monkeypatch.setattr("app.rag.llm.time.sleep", lambda s: None)
    s = Settings(llm_provider="groq", groq_api_key="k", llm_max_retries=2)
    bad = LLMFactory.create(s, transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(LLMConfigurationError):
        bad.generate([{"role": "user", "content": "x"}])
    down = LLMFactory.create(s, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(LLMError):
        down.generate([{"role": "user", "content": "x"}])
