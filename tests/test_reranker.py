from app.config import Settings
from app.rag.reranker import CrossEncoderReranker, NoOpReranker, create_reranker
from app.rag.vectorstore import RetrievedChunk


def _c(i, score, text):
    return RetrievedChunk(id=str(i), text=text, url=f"u{i}", title="t", heading="", section="s", score=score, retrieval_score=score)


def test_noop_orders_by_score():
    out = NoOpReranker().rerank("q", [_c(1, 0.2, "a"), _c(2, 0.9, "b"), _c(3, 0.5, "c")], top_n=2)
    assert [c.id for c in out] == ["2", "3"]


def test_cross_encoder_reorders_and_keeps_retrieval_score():
    scorer = lambda q, docs: [10.0 if "cdt" in d else 0.0 for d in docs]
    rr = CrossEncoderReranker("x", scorer=scorer)
    out = rr.rerank("cdt", [_c(1, 0.9, "cuenta"), _c(2, 0.1, "cdt tasa")], top_n=1)
    assert out[0].id == "2" and out[0].score == 10.0 and out[0].retrieval_score == 0.1


def test_factory_disabled_and_fallback(monkeypatch):
    assert isinstance(create_reranker(Settings(reranker_enabled=False)), NoOpReranker)

    def boom(*a, **k):
        raise RuntimeError("sin red")

    monkeypatch.setattr(CrossEncoderReranker, "__init__", boom)
    assert isinstance(create_reranker(Settings(reranker_enabled=True)), NoOpReranker)
