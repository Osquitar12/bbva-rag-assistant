"""Indexación: `python -m app.ingestion.run [--reindex]`.

También existe `python -m app.ingestion.run --all` que ejecuta scraping + indexación
(es lo que corre el servicio `ingest` de docker-compose).
"""
from __future__ import annotations

import argparse
import logging
import time

from app.config import get_settings
from app.ingestion.chunker import StructuralChunker
from app.logging_config import setup_logging
from app.rag.embeddings import EmbeddingProvider, create_embedder
from app.rag.vectorstore import QdrantVectorStore
from app.scraper.storage import LocalStorage

logger = logging.getLogger("ingestion")


def index_documents(
    storage: LocalStorage,
    chunker: StructuralChunker,
    embedder: EmbeddingProvider,
    store: QdrantVectorStore,
    reindex: bool = False,
) -> int:
    if store.count() > 0 and not reindex:
        logger.info("La colección ya tiene %d vectores; se omite (usa REINDEX=true)", store.count())
        return 0
    docs = storage.load_clean()
    if not docs:
        logger.warning("No hay documentos limpios para indexar. Ejecuta primero el scraper.")
        return 0
    chunks = [c for d in docs for c in chunker.split(d)]
    logger.info("%d documentos -> %d chunks", len(docs), len(chunks))
    if reindex:
        store.recreate(embedder.dimension)
    else:
        store.ensure_collection(embedder.dimension)
    t0 = time.perf_counter()
    vectors = embedder.embed_documents([c.embedding_text for c in chunks])
    store.upsert(chunks, vectors)
    logger.info("Indexados %d chunks en %.1fs", len(chunks), time.perf_counter() - t0)
    return len(chunks)


def wait_for_qdrant(store: QdrantVectorStore, attempts: int = 30) -> None:
    for i in range(attempts):
        try:
            store.client.get_collections()
            return
        except Exception:
            logger.info("Esperando a Qdrant (%d/%d)...", i + 1, attempts)
            time.sleep(2)
    raise RuntimeError("Qdrant no está disponible")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--all", action="store_true", help="scraping + indexación")
    args = parser.parse_args()
    s = get_settings()

    if args.all:
        from app.scraper.run import run_scraping

        run_scraping()

    store = QdrantVectorStore.from_url(s.qdrant_url, s.qdrant_collection)
    wait_for_qdrant(store)
    index_documents(
        LocalStorage(s.raw_dir, s.clean_dir),
        StructuralChunker(s.chunk_size, s.chunk_overlap, s.chunk_min_chars),
        create_embedder(s),
        store,
        reindex=args.reindex or s.reindex,
    )


if __name__ == "__main__":
    main()
