from fastapi.testclient import TestClient

from app.api.main import create_app
from app.exceptions import LLMError


def _client(service):
    return TestClient(create_app(lambda: service))


def test_chat_flow_history_feedback_metrics(rag_service):
    with _client(rag_service) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/chat", json={"message": "¿Qué es un CDT inversión?"}).json()
        assert r["answered"] and r["sources"]
        sid = r["session_id"]
        r2 = c.post("/chat", json={"message": "¿y la tarjeta Visa Aqua requisitos?", "session_id": sid}).json()
        assert r2["session_id"] == sid
        assert len(c.get(f"/sessions/{sid}/history").json()) == 4
        assert c.get("/sessions").json()[0]["id"] == sid
        assert c.post("/feedback", json={"interaction_id": r["interaction_id"], "value": 1}).json() == {"ok": True}
        assert c.post("/feedback", json={"interaction_id": 999, "value": 1}).status_code == 404
        assert c.post("/feedback", json={"interaction_id": r["interaction_id"], "value": 3}).status_code == 422
        m = c.get("/metrics").json()
        assert m["usage"]["total_questions"] == 2 and m["quality"]["satisfaction_rate"] == 1.0


def test_validation_and_llm_errors(rag_service):
    with _client(rag_service) as c:
        assert c.post("/chat", json={"message": ""}).status_code == 422

        def boom(*a, **k):
            raise LLMError("caído")

        rag_service.llm.generate = boom
        resp = c.post("/chat", json={"message": "CDT"})
        assert resp.status_code == 503 and "caído" in resp.json()["detail"]


def test_startup_failure_reports_degraded():
    def broken():
        raise RuntimeError("Falta GROQ_API_KEY")

    with TestClient(create_app(broken)) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and "GROQ_API_KEY" in h["startup_error"]
        assert c.post("/chat", json={"message": "hola"}).status_code == 503
