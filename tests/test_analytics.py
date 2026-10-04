from app.analytics.cli import render
from app.analytics.metrics import ConversationAnalytics, tokenize
from app.memory.repository import InteractionRecord, SQLAlchemyConversationRepository

SRC = [{"url": "https://www.bbva.com.co/personas/productos/tarjetas/credito/visa/aqua.html", "title": "Aqua"}]


def _repo():
    repo = SQLAlchemyConversationRepository("sqlite://")
    for sid in ("s1", "s2"):
        repo.get_or_create_session(sid)
    repo.log_interaction(InteractionRecord("s1", "¿Requisitos de la tarjeta Aqua?", "x [1]", True, sources=SRC,
                                           latency_ms=1000, top_score=0.8))
    iid = repo.log_interaction(InteractionRecord("s1", "¿y la cuota de manejo de la tarjeta?", "y", True, sources=SRC,
                                                 rewritten_query="cuota manejo Aqua", latency_ms=2000, top_score=0.6))
    repo.log_interaction(InteractionRecord("s2", "¿Quién ganó el mundial?", "No encontré", False, latency_ms=500))
    repo.log_interaction(InteractionRecord("s2", "CDT", "", False, error="timeout"))
    repo.set_feedback(iid, 1)
    return repo


def test_metrics_values():
    m = ConversationAnalytics(_repo(), minutes_saved_per_answer=6).compute()
    assert m["usage"]["total_sessions"] == 2 and m["usage"]["total_questions"] == 4
    assert m["quality"]["answer_rate"] == 0.5
    assert m["quality"]["error_rate"] == 0.25
    assert m["quality"]["no_answer_rate"] == 0.25
    assert m["quality"]["satisfaction_rate"] == 1.0
    assert m["impact"]["estimated_hours_saved"] == 0.2
    assert m["content"]["top_sections"][0] == ("personas/productos/tarjetas", 2)
    assert ("tarjeta", 2) in m["content"]["top_terms"]
    assert m["gaps"]["unanswered_questions"] == ["¿Quién ganó el mundial?"]
    assert m["usage"]["follow_up_rate"] == round(1 / 3, 4)
    assert m["performance"]["latency_p50_ms"] == 1000


def test_empty_history_and_render():
    m = ConversationAnalytics(SQLAlchemyConversationRepository("sqlite://")).compute()
    assert m["usage"]["total_questions"] == 0 and m["quality"]["answer_rate"] == 0.0
    assert "REPORTE" in render(m)
    assert "Tiempo ahorrado" in render(ConversationAnalytics(_repo()).compute())


def test_tokenize_removes_stopwords_and_accents():
    assert tokenize("¿Cuáles son los requisitos del crédito?") == ["requisitos", "credito"]
