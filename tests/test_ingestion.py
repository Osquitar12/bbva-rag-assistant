from qdrant_client import QdrantClient

from app.ingestion.chunker import StructuralChunker
from app.ingestion.run import index_documents
from app.rag.vectorstore import QdrantVectorStore
from app.scraper.storage import CleanDocument, LocalStorage


def _doc(text: str, url: str = "https://x.co/p.html") -> CleanDocument:
    return CleanDocument(url=url, title="Producto", description="", section="personas", text=text)


def test_chunker_respects_size_and_headings():
    body = " ".join(f"Frase número {i} sobre la tarjeta de crédito." for i in range(120))
    text = f"# Tarjeta\nIntro de la tarjeta con suficiente texto para ser una sección propia del documento.\n## Beneficios\n{body}"
    chunks = StructuralChunker(chunk_size=400, chunk_overlap=50, min_chars=20).split(_doc(text))
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 + 60 for c in chunks)
    assert any("Beneficios" in c.heading for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    # ids deterministas => upsert idempotente
    again = StructuralChunker(chunk_size=400, chunk_overlap=50, min_chars=20).split(_doc(text))
    assert [c.id for c in chunks] == [c.id for c in again]


def test_short_sections_keep_their_own_heading_when_merged():
    text = (
        "# Historia\n## Línea de tiempo\n"
        "### 1999\nBBV se fusiona con Argentaria.\n"
        "### 2004\nLa entidad pasa a llamarse BBVA Colombia.\n"
        "##### Requisitos\n" + "Debes ser mayor de edad y tener cédula de ciudadanía colombiana. " * 3
    )
    chunks = StructuralChunker(chunk_size=500, chunk_overlap=50, min_chars=20).split(_doc(text))
    timeline = next(c for c in chunks if "2004" in c.text)
    assert timeline.heading == "Historia > Línea de tiempo"
    assert "1999\nBBV se fusiona con Argentaria.\n2004\nLa entidad pasa a llamarse BBVA Colombia." in timeline.text
    assert any(c.heading.endswith("Requisitos") for c in chunks)  # h5 también es sección


def test_overlap_shares_text_between_consecutive_chunks():
    text = " ".join(f"palabra{i}" for i in range(400))
    chunks = StructuralChunker(chunk_size=300, chunk_overlap=60, min_chars=10).split(_doc(text))
    first_tail = chunks[0].text.split()[-1]
    assert first_tail in chunks[1].text


def test_index_and_search(tmp_path, fake_embedder):
    storage = LocalStorage(tmp_path / "raw", tmp_path / "clean")
    storage.save_clean(_doc("# CDT\nEl CDT es una inversión a término fijo con tasa garantizada.", "https://x.co/cdt.html"))
    storage.save_clean(_doc("# Cuenta\nLa cuenta de ahorros digital se abre desde la app sin ir a la oficina.", "https://x.co/cuenta.html"))
    store = QdrantVectorStore(QdrantClient(":memory:"), "test")
    n = index_documents(storage, StructuralChunker(500, 50, 10), fake_embedder, store)
    assert n == 2 and store.count() == 2
    # segunda ejecución no duplica
    assert index_documents(storage, StructuralChunker(500, 50, 10), fake_embedder, store) == 0
    hits = store.search(fake_embedder.embed_query("inversión CDT tasa"), top_k=1)
    assert hits[0].url == "https://x.co/cdt.html"
