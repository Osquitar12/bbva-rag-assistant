"""Rerankers (Strategy): reordenan los candidatos recuperados antes de pasarlos al LLM."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from app.analytics.metrics import tokenize
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


class KeywordReranker(Reranker):
    """Reranker léxico sin modelo: suma a la similitud vectorial un bono por los
    términos de la pregunta que aparecen en el chunk, ponderados por lo raros que son
    entre los candidatos (IDF). Rescata coincidencias exactas que los embeddings
    diluyen, como un año ("2004") o el título de una sección ("Requisitos")."""

    name = "keyword"

    def __init__(self, boost: float = 0.2):
        self.boost = boost

    def rerank(self, query, chunks, top_n):
        terms = set(tokenize(query))
        if not chunks or not terms:
            return NoOpReranker().rerank(query, chunks, top_n)
        heads = [set(tokenize(c.heading)) for c in chunks]
        bodies = [set(tokenize(c.text)) for c in chunks]
        idf = {}
        for term in terms:
            df = sum(term in head or term in body for head, body in zip(heads, bodies))
            if df:  # un término que no está en ningún candidato no discrimina
                idf[term] = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
        total = sum(idf.values())
        if not total:
            return NoOpReranker().rerank(query, chunks, top_n)

        def bonus(head: set[str], body: set[str]) -> float:
            # Coincidir con el título de la sección pesa el doble que con el cuerpo
            return sum(w * (2 if t in head else 1 if t in body else 0) for t, w in idf.items()) / total

        rescored = [
            replace(c, score=c.score + self.boost * bonus(head, body))
            for c, head, body in zip(chunks, heads, bodies)
        ]
        return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_n]


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
        return KeywordReranker()
    try:
        return CrossEncoderReranker(settings.reranker_model, settings.model_cache_dir)
    except Exception as exc:  # sin disco/red: degradar sin romper el sistema
        logger.warning("No se pudo cargar el reranker (%s); se usará KeywordReranker", exc)
        return KeywordReranker()
