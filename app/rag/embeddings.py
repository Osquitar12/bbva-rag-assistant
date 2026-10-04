"""Proveedores de embeddings (Strategy) y su fábrica (Factory)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider(EmbeddingProvider):
    """Embeddings locales vía ONNX (fastembed): sin PyTorch, imagen Docker liviana."""

    def __init__(self, model_name: str, cache_dir: Path | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding  # import perezoso: carga pesada

        logger.info("Cargando modelo de embeddings %s", model_name)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self._batch_size = batch_size
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def create_embedder(settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == "fastembed":
        return FastEmbedProvider(settings.embedding_model, settings.model_cache_dir)
    raise ValueError(f"Proveedor de embeddings no soportado: {provider}")
