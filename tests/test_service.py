import pytest

from app.exceptions import LLMError
from app.rag.service import is_no_answer


def test_ask_answers_with_sources_and_persists(rag_service):
    r = rag_service.ask("¿Qué es un CDT inversión plazo fijo?")
    assert r.answered and r.sources and r.sources[0].url.endswith("cdt.html")
    assert r.rewritten_query is None  # primer turno: sin reformulación
    hist = rag_service.repo.get_history(r.session_id)
    assert [m.role for m in hist] == ["user", "assistant"]


def test_follow_up_uses_history_and_rewrite(rag_service):
    r1 = rag_service.ask("Háblame de la tarjeta Visa Aqua")
    r2 = rag_service.ask("¿y qué requisitos pide?", session_id=r1.session_id)
    assert r2.rewritten_query == "requisitos tarjeta Visa Aqua"
    last_call = rag_service.llm.calls[-1]["messages"]
    # system + 2 mensajes de historial + pregunta actual
    assert len(last_call) == 4 and last_call[1]["content"] == "Háblame de la tarjeta Visa Aqua"


def test_history_window_is_respected(rag_service):
    sid = "fija"
    for i in range(5):
        rag_service.ask(f"pregunta CDT {i}", session_id=sid)
    msgs = rag_service.llm.calls[-1]["messages"]
    assert len(msgs) == 1 + rag_service.s.history_window_n + 1


def test_no_context_marks_unanswered(rag_service):
    rag_service.s.min_relevance_score = 0.99
    r = rag_service.ask("¿Quién ganó el mundial?")
    assert not r.answered and r.sources == []


def test_llm_error_is_logged_and_raised(rag_service):
    def boom(*a, **k):
        raise LLMError("caído")

    rag_service.llm.generate = boom
    with pytest.raises(LLMError):
        rag_service.ask("CDT", session_id="e")
    rec = list(rag_service.repo.iter_interactions())[-1]
    assert rec.error == "caído" and rag_service.repo.get_history("e") == []


def test_empty_question_rejected(rag_service):
    with pytest.raises(ValueError):
        rag_service.ask("   ")


def test_is_no_answer_accent_insensitive():
    assert is_no_answer("no encontre esa informacion en el sitio")
    assert not is_no_answer("El CDT paga intereses")
