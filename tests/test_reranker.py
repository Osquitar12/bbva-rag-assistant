from app.config import Settings
from app.rag.reranker import CrossEncoderReranker, KeywordReranker, NoOpReranker, create_reranker
from app.rag.vectorstore import RetrievedChunk


def _c(i, score, text):
    return RetrievedChunk(id=str(i), text=text, url=f"u{i}", title="t", heading="", section="s", score=score, retrieval_score=score)


def test_noop_orders_by_score():
    out = NoOpReranker().rerank("q", [_c(1, 0.2, "a"), _c(2, 0.9, "b"), _c(3, 0.5, "c")], top_n=2)
    assert [c.id for c in out] == ["2", "3"]


def test_keyword_reranker_promotes_exact_matches():
    chunks = [_c(1, 0.75, "La historia comienza cuando BBV compra el Banco Ganadero"),
              _c(2, 0.74, "BBVA cumple 30 años en Colombia"),
              _c(3, 0.70, "2004 La entidad pasa a llamarse BBVA Colombia")]
    out = KeywordReranker().rerank("¿Qué pasó en 2004?", chunks, top_n=2)
    assert out[0].id == "3" and out[0].retrieval_score == 0.70
    # sin términos útiles conserva el orden vectorial
    assert [c.id for c in KeywordReranker().rerank("¿y?", chunks, top_n=2)] == ["1", "2"]


def test_cross_encoder_reorders_and_keeps_retrieval_score():
    scorer = lambda q, docs: [10.0 if "cdt" in d else 0.0 for d in docs]
    rr = CrossEncoderReranker("x", scorer=scorer)
    out = rr.rerank("cdt", [_c(1, 0.9, "cuenta"), _c(2, 0.1, "cdt tasa")], top_n=1)
    assert out[0].id == "2" and out[0].score == 10.0 and out[0].retrieval_score == 0.1


def test_factory_disabled_and_fallback(monkeypatch):
    assert isinstance(create_reranker(Settings(reranker_enabled=False)), KeywordReranker)

    def boom(*a, **k):
        raise RuntimeError("sin red")

    monkeypatch.setattr(CrossEncoderReranker, "__init__", boom)
    assert isinstance(create_reranker(Settings(reranker_enabled=True)), KeywordReranker)
