"""Patrón Repository: el dominio (RAGService, analítica) no conoce SQL ni SQLAlchemy.

Cambiar SQLite por Postgres solo requiere otra DATABASE_URL; cambiar a Redis o
Mongo, otra implementación de `ConversationRepository`.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.models import Base, ChatMessage, ChatSession, InteractionLog, utcnow


@dataclass(frozen=True)
class MessageDTO:
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class SessionSummary:
    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    preview: str


@dataclass
class InteractionRecord:
    session_id: str
    question: str
    answer: str
    answered: bool
    rewritten_query: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    top_score: float | None = None
    num_chunks: int = 0
    sources: list[dict] | None = None
    llm_provider: str = ""
    llm_model: str = ""
    feedback: int | None = None
    created_at: datetime | None = None
    id: int | None = None


class ConversationRepository(ABC):
    @abstractmethod
    def get_or_create_session(self, session_id: str) -> str: ...

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    def get_last_messages(self, session_id: str, n: int) -> list[MessageDTO]: ...

    @abstractmethod
    def get_history(self, session_id: str) -> list[MessageDTO]: ...

    @abstractmethod
    def list_sessions(self, limit: int = 50) -> list[SessionSummary]: ...

    @abstractmethod
    def log_interaction(self, record: InteractionRecord) -> int: ...

    @abstractmethod
    def set_feedback(self, interaction_id: int, value: int) -> bool: ...

    @abstractmethod
    def iter_interactions(self) -> Iterator[InteractionRecord]: ...


class SQLAlchemyConversationRepository(ConversationRepository):
    def __init__(self, database_url: str):
        kwargs: dict = {}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if database_url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
            else:
                Path(database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, **kwargs)
        if database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", lambda c, _: c.execute("PRAGMA journal_mode=WAL"))
        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def _db(self) -> Session:
        return self._sessions()

    def get_or_create_session(self, session_id: str) -> str:
        with self._db() as db, db.begin():
            if db.get(ChatSession, session_id) is None:
                db.add(ChatSession(id=session_id))
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._db() as db, db.begin():
            sess = db.get(ChatSession, session_id) or ChatSession(id=session_id)
            sess.updated_at = utcnow()
            db.add(sess)
            db.add(ChatMessage(session_id=session_id, role=role, content=content))

    @staticmethod
    def _dto(m: ChatMessage) -> MessageDTO:
        return MessageDTO(m.role, m.content, m.created_at)

    def get_last_messages(self, session_id: str, n: int) -> list[MessageDTO]:
        if n <= 0:
            return []
        with self._db() as db:
            rows = db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id.desc())
                .limit(n)
            ).all()
        return [self._dto(m) for m in reversed(rows)]

    def get_history(self, session_id: str) -> list[MessageDTO]:
        with self._db() as db:
            rows = db.scalars(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
            ).all()
        return [self._dto(m) for m in rows]

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        with self._db() as db:
            counts = (
                select(ChatMessage.session_id, func.count(ChatMessage.id).label("n"))
                .group_by(ChatMessage.session_id)
                .subquery()
            )
            rows = db.execute(
                select(ChatSession, func.coalesce(counts.c.n, 0))
                .outerjoin(counts, counts.c.session_id == ChatSession.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            ).all()
            result = []
            for sess, n in rows:
                first = db.scalars(
                    select(ChatMessage.content)
                    .where(ChatMessage.session_id == sess.id, ChatMessage.role == "user")
                    .order_by(ChatMessage.id)
                    .limit(1)
                ).first()
                result.append(SessionSummary(sess.id, sess.created_at, sess.updated_at, n, (first or "")[:80]))
        return result

    def log_interaction(self, record: InteractionRecord) -> int:
        with self._db() as db, db.begin():
            row = InteractionLog(
                session_id=record.session_id, question=record.question, rewritten_query=record.rewritten_query,
                answer=record.answer, answered=record.answered, error=record.error, latency_ms=record.latency_ms,
                retrieval_ms=record.retrieval_ms, generation_ms=record.generation_ms, top_score=record.top_score,
                num_chunks=record.num_chunks, sources_json=json.dumps(record.sources or [], ensure_ascii=False),
                llm_provider=record.llm_provider, llm_model=record.llm_model,
            )
            db.add(row)
            db.flush()
            return row.id

    def set_feedback(self, interaction_id: int, value: int) -> bool:
        if value not in (1, -1):
            raise ValueError("feedback debe ser 1 o -1")
        with self._db() as db, db.begin():
            row = db.get(InteractionLog, interaction_id)
            if row is None:
                return False
            row.feedback = value
            return True

    def iter_interactions(self) -> Iterator[InteractionRecord]:
        with self._db() as db:
            for r in db.scalars(select(InteractionLog).order_by(InteractionLog.id)).yield_per(500):
                yield InteractionRecord(
                    id=r.id, session_id=r.session_id, question=r.question, answer=r.answer, answered=r.answered,
                    rewritten_query=r.rewritten_query, error=r.error, latency_ms=r.latency_ms,
                    retrieval_ms=r.retrieval_ms, generation_ms=r.generation_ms, top_score=r.top_score,
                    num_chunks=r.num_chunks, sources=json.loads(r.sources_json or "[]"),
                    llm_provider=r.llm_provider, llm_model=r.llm_model, feedback=r.feedback, created_at=r.created_at,
                )
