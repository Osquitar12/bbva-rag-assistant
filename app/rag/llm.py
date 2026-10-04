"""Proveedores de LLM intercambiables (Strategy) creados por una fábrica (Factory).

Groq y Ollama exponen una API compatible con OpenAI (`/chat/completions`), así
que comparten una base común y solo difieren en autenticación y URL.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import httpx

from app.exceptions import LLMConfigurationError, LLMError

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


class OpenAICompatibleLLM(LLMProvider):
    _RETRYABLE = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        self.model, self.temperature, self.max_tokens, self.max_retries = model, temperature, max_tokens, max_retries

    def _payload(self, messages, model, temperature, max_tokens) -> dict:
        return {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

    def generate(self, messages, model=None, temperature=None, max_tokens=None) -> str:
        payload = self._payload(messages, model, temperature, max_tokens)
        last_error = "desconocido"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                last_error = "timeout"
            except httpx.HTTPError as exc:
                last_error = f"error de red: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        content = resp.json()["choices"][0]["message"].get("content") or ""
                    except (KeyError, IndexError, ValueError) as exc:
                        raise LLMError(f"Respuesta inesperada del proveedor {self.name}: {exc}") from exc
                    return content.strip()
                if resp.status_code in (401, 403):
                    raise LLMConfigurationError(
                        f"{self.name}: credenciales inválidas (HTTP {resp.status_code}). Revisa tu API key en .env"
                    )
                if resp.status_code not in self._RETRYABLE:
                    raise LLMError(f"{self.name}: HTTP {resp.status_code} - {resp.text[:300]}")
                last_error = f"HTTP {resp.status_code}"
            wait = min(2 ** attempt, 10)
            logger.warning("%s falló (%s), reintento %d/%d en %ss", self.name, last_error, attempt, self.max_retries, wait)
            if attempt < self.max_retries:
                time.sleep(wait)
        raise LLMError(f"{self.name} no respondió tras {self.max_retries} intentos ({last_error})")


class GroqLLM(OpenAICompatibleLLM):
    name = "groq"

    def __init__(self, api_key: str, **kwargs):
        if not api_key:
            raise LLMConfigurationError(
                "Falta GROQ_API_KEY. Crea una gratis en https://console.groq.com/keys y ponla en el archivo .env"
            )
        super().__init__(api_key=api_key, **kwargs)


class OllamaLLM(OpenAICompatibleLLM):
    name = "ollama"


class LLMFactory:
    """Crea el proveedor según LLM_PROVIDER. Nuevos proveedores se registran aquí."""

    _registry: dict[str, type[OpenAICompatibleLLM]] = {"groq": GroqLLM, "ollama": OllamaLLM}

    @classmethod
    def register(cls, name: str, provider: type[OpenAICompatibleLLM]) -> None:
        cls._registry[name] = provider

    @classmethod
    def create(cls, settings, transport: httpx.BaseTransport | None = None) -> LLMProvider:
        name = settings.llm_provider.lower()
        common = dict(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            transport=transport,
        )
        if name == "groq":
            return GroqLLM(api_key=settings.groq_api_key, base_url=settings.groq_base_url, **common)
        if name == "ollama":
            return OllamaLLM(base_url=settings.ollama_base_url, **common)
        if name in cls._registry:
            return cls._registry[name](**common)
        raise LLMConfigurationError(f"LLM_PROVIDER no soportado: {name}. Opciones: {sorted(cls._registry)}")
