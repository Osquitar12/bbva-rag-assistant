"""Acceso a Qdrant (base de datos vectorial self-hosted)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.exceptions import VectorStoreError
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    url: str
    title: str
    heading: str
    section: str
    score: float  # similitud coseno; se sobrescribe con el score del reranker
    retrieval_score: float = 0.0


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client, self.collection = client, collection

    @classmethod
    def from_url(cls, url: str, collection: str) -> "QdrantVectorStore":
        return cls(QdrantClient(url=url, timeout=30), collection)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def recreate(self, dimension: int) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)
        self.ensure_collection(dimension)

    def ensure_collection(self, dimension: int) -> None:
        if self.exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
        )
        self.client.create_payload_index(self.collection, "section", qm.PayloadSchemaType.KEYWORD)
        logger.info("Colección %s creada (dim=%d)", self.collection, dimension)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 128) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks y vectores con longitudes distintas")
        for start in range(0, len(chunks), batch_size):
            points = [
                qm.PointStruct(
                    id=c.id,
                    vector=v,
                    payload={
                        "text": c.text, "url": c.url, "title": c.title, "heading": c.heading,
                        "section": c.section, "chunk_index": c.chunk_index, **c.metadata,
                    },
                )
                for c, v in zip(chunks[start : start + batch_size], vectors[start : start + batch_size])
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        try:
            res = self.client.query_points(self.collection, query=vector, limit=top_k, with_payload=True)
        except Exception as exc:
            raise VectorStoreError(f"Error consultando Qdrant: {exc}") from exc
        out = []
        for p in res.points:
            pl = p.payload or {}
            out.append(
                RetrievedChunk(
                    id=str(p.id), text=pl.get("text", ""), url=pl.get("url", ""), title=pl.get("title", ""),
                    heading=pl.get("heading", ""), section=pl.get("section", ""),
                    score=float(p.score), retrieval_score=float(p.score),
                )
            )
        return out
