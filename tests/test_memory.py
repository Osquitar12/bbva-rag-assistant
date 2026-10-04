import pytest

from app.memory.repository import InteractionRecord, SQLAlchemyConversationRepository


@pytest.fixture
def repo():
    return SQLAlchemyConversationRepository("sqlite://")


def test_window_returns_last_n_in_chronological_order(repo):
    repo.get_or_create_session("s1")
    for i in range(10):
        repo.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
    last = repo.get_last_messages("s1", 4)
    assert [m.content for m in last] == ["m6", "m7", "m8", "m9"]
    assert repo.get_last_messages("s1", 0) == []
    assert len(repo.get_history("s1")) == 10


def test_sessions_are_isolated_and_listed(repo):
    repo.add_message("a", "user", "pregunta A")
    repo.add_message("b", "user", "pregunta B")
    repo.add_message("b", "assistant", "respuesta B")
    assert [m.content for m in repo.get_history("a")] == ["pregunta A"]
    sessions = {s.id: s for s in repo.list_sessions()}
    assert sessions["b"].message_count == 2 and sessions["a"].preview == "pregunta A"


def test_interactions_and_feedback(repo):
    repo.get_or_create_session("s")
    iid = repo.log_interaction(InteractionRecord("s", "q", "a", True, sources=[{"url": "u"}], latency_ms=12.5))
    assert repo.set_feedback(iid, 1) is True
    assert repo.set_feedback(9999, 1) is False
    with pytest.raises(ValueError):
        repo.set_feedback(iid, 5)
    rec = list(repo.iter_interactions())[0]
    assert rec.feedback == 1 and rec.sources == [{"url": "u"}] and rec.latency_ms == 12.5


def test_file_database_persists(tmp_path):
    url = f"sqlite:///{tmp_path}/sub/chat.db"
    SQLAlchemyConversationRepository(url).add_message("s", "user", "hola")
    assert SQLAlchemyConversationRepository(url).get_history("s")[0].content == "hola"
