"""Modelos de persistencia del historial y de la bitácora de interacciones."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    session: Mapped[ChatSession] = relationship(back_populates="messages")


class InteractionLog(Base):
    """Una fila por pregunta respondida: alimenta el módulo de analítica."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str] = mapped_column(Text, default="")
    answered: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0)
    generation_ms: Mapped[float] = mapped_column(Float, default=0.0)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    llm_provider: Mapped[str] = mapped_column(String(32), default="")
    llm_model: Mapped[str] = mapped_column(String(128), default="")
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 | -1
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
