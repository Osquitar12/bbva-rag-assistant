"""Rerankers (Strategy): reordenan los candidatos recuperados antes de pasarlos al LLM."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from app.rag.vectorstore import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    name = "base"

    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


class NoOpReranker(Reranker):
    """Conserva el orden por similitud vectorial y recorta a top_n."""

    name = "none"

    def rerank(self, query, chunks, top_n):
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]


class CrossEncoderReranker(Reranker):
    """Cross-encoder multilingüe (fastembed/ONNX). Evalúa pregunta y pasaje juntos,
    lo que es más preciso que la similitud de embeddings independientes."""

    name = "cross-encoder"

    def __init__(self, model_name: str, cache_dir: Path | None = None, scorer=None):
        if scorer is not None:  # inyección para tests
            self._scorer = scorer
        else:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("Cargando reranker %s", model_name)
            model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)
            self._scorer = lambda q, docs: list(model.rerank(q, docs))

    def rerank(self, query, chunks, top_n):
        if not chunks:
            return []
        scores = self._scorer(query, [f"{c.title}\n{c.text}" for c in chunks])
        rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_n]


def create_reranker(settings) -> Reranker:
    if not settings.reranker_enabled:
        return NoOpReranker()
    try:
        return CrossEncoderReranker(settings.reranker_model, settings.model_cache_dir)
    except Exception as exc:  # sin disco/red: degradar sin romper el sistema
        logger.warning("No se pudo cargar el reranker (%s); se usará NoOpReranker", exc)
        return NoOpReranker()
