"""API REST del asistente: `uvicorn app.api.main:app`."""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.analytics.metrics import ConversationAnalytics
from app.config import get_settings
from app.exceptions import LLMConfigurationError, RAGError
from app.logging_config import setup_logging
from app.rag.service import RAGService, build_rag_service

logger = logging.getLogger("api")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, examples=["¿Qué beneficios tiene la tarjeta Visa Aqua?"])
    session_id: str | None = Field(default=None, max_length=64, description="Si se omite, se crea una sesión nueva")


class SourceOut(BaseModel):
    title: str
    url: str
    score: float


class ChatResponseOut(BaseModel):
    session_id: str
    answer: str
    answered: bool
    sources: list[SourceOut]
    rewritten_query: str | None
    interaction_id: int | None
    latency_ms: float


class FeedbackRequest(BaseModel):
    interaction_id: int
    value: int = Field(..., description="1 = útil, -1 = no útil")


def create_app(service_factory: Callable[[], RAGService] | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        app.state.service, app.state.startup_error = None, None
        try:
            app.state.service = (service_factory or (lambda: build_rag_service(settings)))()
            logger.info("RAGService listo (LLM=%s, modelo=%s)", settings.llm_provider, settings.llm_model)
        except Exception as exc:  # la API arranca igual y explica el problema en /health y /chat
            app.state.startup_error = str(exc)
            logger.error("No se pudo inicializar el servicio RAG: %s", exc)
        yield

    app = FastAPI(
        title="BBVA Colombia RAG Assistant",
        description="Asistente conversacional sobre el contenido público de bbva.com.co",
        version="1.0.0",
        lifespan=lifespan,
    )

    def service(request: Request) -> RAGService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(503, f"Servicio no inicializado: {request.app.state.startup_error}")
        return svc

    @app.exception_handler(LLMConfigurationError)
    async def _config_error(_: Request, exc: LLMConfigurationError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(RAGError)
    async def _rag_error(_: Request, exc: RAGError):
        return JSONResponse(
            status_code=503,
            content={"detail": f"El asistente no está disponible temporalmente: {exc}"},
        )

    @app.get("/health")
    def health(request: Request):
        svc = request.app.state.service
        info = {"status": "ok" if svc else "degraded", "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model, "startup_error": request.app.state.startup_error}
        if svc:
            try:
                info["vectors"] = svc.store.count()
                info["reranker"] = svc.reranker.name
            except Exception as exc:
                info.update(status="degraded", vector_store_error=str(exc))
        return info

    @app.post("/chat", response_model=ChatResponseOut)
    def chat(body: ChatRequest, request: Request):
        try:
            r = service(request).ask(body.message, body.session_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return ChatResponseOut(
            session_id=r.session_id, answer=r.answer, answered=r.answered,
            sources=[SourceOut(**s.__dict__) for s in r.sources], rewritten_query=r.rewritten_query,
            interaction_id=r.interaction_id, latency_ms=r.latency_ms,
        )

    @app.get("/sessions")
    def sessions(request: Request, limit: int = 50):
        return [s.__dict__ for s in service(request).repo.list_sessions(limit)]

    @app.get("/sessions/{session_id}/history")
    def history(session_id: str, request: Request):
        return [m.__dict__ for m in service(request).repo.get_history(session_id)]

    @app.post("/feedback")
    def feedback(body: FeedbackRequest, request: Request):
        try:
            found = service(request).repo.set_feedback(body.interaction_id, body.value)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not found:
            raise HTTPException(404, "Interacción no encontrada")
        return {"ok": True}

    @app.get("/metrics")
    def metrics(request: Request):
        svc = service(request)
        return ConversationAnalytics(svc.repo, settings.minutes_saved_per_answer).compute()

    return app


app = create_app()
