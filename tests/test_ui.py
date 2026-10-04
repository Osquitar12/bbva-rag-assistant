from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py"


def test_ui_renders_without_api(monkeypatch):
    """Smoke test: la UI compila y muestra errores amigables si la API no responde."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(UI), default_timeout=30).run()
    assert not at.exception
    assert any("No se pudo conectar con la API" in e.value for e in at.error)
