import hashlib
import math

import pytest

from app.rag.embeddings import EmbeddingProvider


class FakeEmbedder(EmbeddingProvider):
    """Embedder determinista basado en bolsa de palabras hasheada (sin descargas)."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


class FakeLLM:
    """LLM falso: responde con el contexto recibido o con 'no encontré'."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, messages, model=None, temperature=None, max_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        content = messages[-1]["content"]
        if "Consulta autónoma" in content:
            return "requisitos tarjeta Visa Aqua"
        if "(sin contexto relevante)" in content:
            return "No encontré esa información en el sitio de BBVA Colombia."
        return "Respuesta basada en el contexto [1]."


@pytest.fixture
def rag_service(tmp_path, fake_embedder):
    from qdrant_client import QdrantClient

    from app.config import Settings
    from app.ingestion.chunker import StructuralChunker
    from app.ingestion.run import index_documents
    from app.memory.repository import SQLAlchemyConversationRepository
    from app.rag.reranker import NoOpReranker
    from app.rag.service import RAGService
    from app.rag.vectorstore import QdrantVectorStore
    from app.scraper.storage import CleanDocument, LocalStorage

    storage = LocalStorage(tmp_path / "raw", tmp_path / "clean")
    docs = {
        "aqua": "# Visa Aqua\nLa tarjeta Visa Aqua tiene CVV dinámico. Requisitos: ser mayor de edad e ingresos de 1 SMMLV.",
        "cdt": "# CDT\nEl CDT es una inversión a plazo fijo con tasa fija garantizada desde 30 días.",
    }
    for slug, text in docs.items():
        storage.save_clean(CleanDocument(f"https://www.bbva.com.co/{slug}.html", slug.upper(), "", "personas", text))
    store = QdrantVectorStore(QdrantClient(":memory:"), "t")
    index_documents(storage, StructuralChunker(500, 50, 10), fake_embedder, store)
    settings = Settings(history_window_n=4, retrieval_top_k=5, rerank_top_n=2, min_relevance_score=0.05)
    return RAGService(fake_embedder, store, NoOpReranker(), FakeLLM(), SQLAlchemyConversationRepository("sqlite://"), settings)
