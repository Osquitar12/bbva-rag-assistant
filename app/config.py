"""Configuración centralizada (externalizada vía variables de entorno / .env).

Patrón Singleton: `get_settings()` está cacheado con lru_cache, de modo que
toda la aplicación comparte una única instancia de configuración.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Rutas ---
    data_dir: Path = Path("data")

    # --- Scraping ---
    scrape_base_url: str = "https://www.bbva.com.co"
    scrape_sitemap_url: str = "https://www.bbva.com.co/sitemap.xml"
    scrape_max_pages: int = 300
    scrape_concurrency: int = 4
    scrape_delay_seconds: float = 0.5
    scrape_timeout_seconds: float = 20.0
    # UA de navegador + identificador propio: algunos WAF bloquean UAs no estándar
    scrape_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36 BBVA-RAG-Assessment/1.0"
    )
    # Patrones de URL (substring) a incluir / excluir, separados por coma
    scrape_include_patterns: str = ""
    scrape_exclude_patterns: str = "/investor-relations/,/herramientas/,/landing/formulario"
    scrape_force: bool = False

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_min_chars: int = 80

    # --- Embeddings / Vector DB ---
    embedding_provider: str = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "bbva_co"
    reindex: bool = False
    model_cache_dir: Path = Path(".cache/models")

    # --- Recuperación ---
    retrieval_top_k: int = 20
    rerank_top_n: int = 4
    reranker_enabled: bool = True
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    # Umbral mínimo de score para usar un chunk. Vacío = sin umbral (los scores del
    # cross-encoder son logits y pueden ser negativos, por eso no hay umbral por defecto)
    min_relevance_score: float | None = Field(default=None)

    # --- LLM ---
    llm_provider: str = "groq"  # groq | ollama
    llm_model: str = "openai/gpt-oss-120b"
    llm_rewrite_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500
    llm_reasoning_effort: str = "low"  # solo modelos gpt-oss en Groq
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://ollama:11434/v1"

    # --- Conversación ---
    history_window_n: int = 6
    query_rewrite_enabled: bool = True
    database_url: str = ""  # por defecto sqlite en data_dir/chat.db

    # --- Analítica ---
    minutes_saved_per_answer: float = 4.0

    # --- Servicios ---
    api_url: str = "http://localhost:8000"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'chat.db').as_posix()}"

    @staticmethod
    def _split(value: str) -> list[str]:
        return [p.strip() for p in value.split(",") if p.strip()]

    @property
    def include_patterns(self) -> list[str]:
        return self._split(self.scrape_include_patterns)

    @property
    def exclude_patterns(self) -> list[str]:
        return self._split(self.scrape_exclude_patterns)


@lru_cache
def get_settings() -> Settings:
    return Settings()
