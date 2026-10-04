"""RAGService: Facade que orquesta todo el flujo conversacional.

    historial -> reformulación -> embedding -> Qdrant (top-K) -> reranker (top-N)
              -> prompt -> LLM -> persistencia (mensajes + bitácora de analítica)

La API y la UI solo conocen `ask()`; los detalles (proveedores, base vectorial,
repositorio) quedan encapsulados y se inyectan, lo que facilita las pruebas.
"""
from __future__ import annotations

import logging
import time
import unicodedata
import uuid
from dataclasses import dataclass, field

from app.exceptions import LLMConfigurationError, RAGError
from app.memory.repository import ConversationRepository, InteractionRecord, MessageDTO
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import LLMProvider
from app.rag.prompts import NO_ANSWER, REWRITE_PROMPT, SYSTEM_PROMPT, build_user_prompt
from app.rag.reranker import Reranker
from app.rag.vectorstore import QdrantVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class Source:
    title: str
    url: str
    score: float


@dataclass
class ChatResponse:
    session_id: str
    answer: str
    answered: bool
    sources: list[Source] = field(default_factory=list)
    rewritten_query: str | None = None
    interaction_id: int | None = None
    latency_ms: float = 0.0


def _normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def is_no_answer(answer: str) -> bool:
    return _normalize("No encontré esa información") in _normalize(answer)


class RAGService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: QdrantVectorStore,
        reranker: Reranker,
        llm: LLMProvider,
        repository: ConversationRepository,
        settings,
    ):
        self.embedder, self.store, self.reranker = embedder, store, reranker
        self.llm, self.repo, self.s = llm, repository, settings

    # ---------- pasos ----------
    def _rewrite(self, question: str, history: list[MessageDTO]) -> str:
        if not (self.s.query_rewrite_enabled and history):
            return question
        convo = "\n".join(f"{m.role}: {m.content[:500]}" for m in history)
        try:
            rewritten = self.llm.generate(
                [{"role": "user", "content": REWRITE_PROMPT.format(history=convo, question=question)}],
                model=self.s.llm_rewrite_model,
                temperature=0.0,
                max_tokens=300,
            )
        except LLMConfigurationError:
            raise
        except RAGError as exc:
            logger.warning("Fallo la reformulación (%s); se usa la pregunta original", exc)
            return question
        rewritten = rewritten.strip().strip('"').splitlines()[0] if rewritten.strip() else ""
        return rewritten if 3 <= len(rewritten) <= 400 else question

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        candidates = self.store.search(self.embedder.embed_query(query), self.s.retrieval_top_k)
        ranked = self.reranker.rerank(query, candidates, self.s.rerank_top_n)
        threshold = self.s.min_relevance_score
        return ranked if threshold is None else [c for c in ranked if c.score >= threshold]

    @staticmethod
    def _sources(chunks: list[RetrievedChunk]) -> list[Source]:
        seen: dict[str, Source] = {}
        for c in chunks:
            if c.url not in seen:
                seen[c.url] = Source(title=c.title or c.url, url=c.url, score=round(c.score, 4))
        return list(seen.values())

    # ---------- API pública ----------
    def ask(self, question: str, session_id: str | None = None) -> ChatResponse:
        question = (question or "").strip()
        if not question:
            raise ValueError("La pregunta no puede estar vacía")
        session_id = session_id or str(uuid.uuid4())
        self.repo.get_or_create_session(session_id)
        history = self.repo.get_last_messages(session_id, self.s.history_window_n)

        t0 = time.perf_counter()
        record = InteractionRecord(
            session_id=session_id, question=question, answer="", answered=False,
            llm_provider=getattr(self.llm, "name", ""), llm_model=self.s.llm_model,
        )
        try:
            query = self._rewrite(question, history)
            record.rewritten_query = query if query != question else None
            t1 = time.perf_counter()
            chunks = self.retrieve(query)
            t2 = time.perf_counter()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += [{"role": m.role, "content": m.content} for m in history]
            messages.append({"role": "user", "content": build_user_prompt(question, chunks)})
            answer = self.llm.generate(messages) or NO_ANSWER
            t3 = time.perf_counter()
        except RAGError as exc:
            record.error = str(exc)
            record.latency_ms = (time.perf_counter() - t0) * 1000
            self.repo.log_interaction(record)
            logger.error("Error respondiendo en sesión %s: %s", session_id, exc)
            raise

        sources = self._sources(chunks)
        record.answer = answer
        record.answered = bool(chunks) and not is_no_answer(answer)
        record.retrieval_ms = (t2 - t1) * 1000
        record.generation_ms = (t3 - t2) * 1000
        record.latency_ms = (t3 - t0) * 1000
        record.top_score = max((c.score for c in chunks), default=None)
        record.num_chunks = len(chunks)
        record.sources = [s.__dict__ for s in sources]

        self.repo.add_message(session_id, "user", question)
        self.repo.add_message(session_id, "assistant", answer)
        interaction_id = self.repo.log_interaction(record)

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            answered=record.answered,
            sources=sources if record.answered else [],
            rewritten_query=record.rewritten_query,
            interaction_id=interaction_id,
            latency_ms=round(record.latency_ms, 1),
        )


def build_rag_service(settings) -> RAGService:
    """Composition root: arma el servicio con las implementaciones configuradas."""
    from app.memory.repository import SQLAlchemyConversationRepository
    from app.rag.embeddings import create_embedder
    from app.rag.llm import LLMFactory
    from app.rag.reranker import create_reranker

    return RAGService(
        embedder=create_embedder(settings),
        store=QdrantVectorStore.from_url(settings.qdrant_url, settings.qdrant_collection),
        reranker=create_reranker(settings),
        llm=LLMFactory.create(settings),
        repository=SQLAlchemyConversationRepository(settings.resolved_database_url),
        settings=settings,
    )
