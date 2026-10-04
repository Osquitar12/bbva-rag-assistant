"""UI minimalista: `streamlit run app/ui/streamlit_app.py`. Solo habla con la API REST."""
from __future__ import annotations

import html
import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=5.0)

st.set_page_config(page_title="Asistente BBVA Colombia", page_icon="💬", layout="wide")


# ---------------- cliente API ----------------
def api(method: str, path: str, **kwargs):
    try:
        resp = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"No se pudo conectar con la API ({API_URL}): {exc}"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"Error {resp.status_code}: {detail}"
    return resp.json(), None


def bar_list(items: list[tuple], max_label_width: int = 160) -> None:
    """Barras horizontales en HTML puro (sin pandas, que no carga en este entorno)."""
    if not items:
        return
    max_v = max((v for _, v in items), default=0) or 1
    rows = []
    for label, value in items:
        label = html.escape(str(label), quote=True)  # el contenido viene de preguntas y del sitio
        pct = max(2, round(value / max_v * 100))
        rows.append(
            "<div style='display:flex;align-items:center;gap:8px;margin:4px 0;font-size:0.85em;'>"
            f"<div style='width:{max_label_width}px;text-align:right;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;' title='{label}'>{label}</div>"
            "<div style='flex:1;background:rgba(127,127,127,0.2);border-radius:4px;'>"
            f"<div style='background:#1f77b4;width:{pct}%;border-radius:4px;padding:2px 6px;"
            f"color:white;white-space:nowrap;'>{value}</div></div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def simple_table(rows: list[tuple], headers: tuple[str, str]) -> None:
    """Tabla en HTML puro (sin pandas)."""
    if not rows:
        return

        for row in rows
    )
    st.markdown(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", unsafe_allow_html=True)


def load_session(session_id: str) -> None:
    st.session_state.session_id = session_id
    history, err = api("GET", f"/sessions/{session_id}/history")
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "sources": [], "interaction_id": None}
        for m in (history or [])
    ]
    if err:
        st.session_state.error = err


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.feedback_sent = set()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Sesión")
    st.caption("El historial se guarda por ID. Reutiliza un ID para continuar una conversación.")
    st.code(st.session_state.session_id, language=None)
    if st.button("➕ Nueva conversación", width="stretch"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    sessions, _ = api("GET", "/sessions", params={"limit": 30})
    if sessions:
        labels = {s["id"]: f"{(s['preview'] or 'sin mensajes')[:40]} · {s['message_count']} msgs" for s in sessions}
        chosen = st.selectbox("Conversaciones anteriores", options=[""] + list(labels), format_func=lambda x: labels.get(x, "—"))
        if chosen and chosen != st.session_state.session_id:
            load_session(chosen)
            st.rerun()

    manual = st.text_input("…o pega un ID de sesión")
    if manual and manual.strip() != st.session_state.session_id:
        load_session(manual.strip())
        st.rerun()

    st.divider()
    health, err = api("GET", "/health")
    if health:
        ok = health.get("status") == "ok"
        st.markdown(f"**Estado:** {'🟢 operativo' if ok else '🟠 degradado'}")
        st.caption(
            f"LLM: {health.get('llm_provider')} · {health.get('llm_model')}  \n"
            f"Vectores indexados: {health.get('vectors', '—')}  \nReranker: {health.get('reranker', '—')}"
        )
        if health.get("startup_error"):
            st.error(health["startup_error"])
    else:
        st.error(err)

tab_chat, tab_metrics = st.tabs(["💬 Chat", "📊 Métricas"])

# ---------------- chat ----------------
with tab_chat:
    st.title("Asistente de consulta · bbva.com.co")
    st.caption("Responde con la información publicada en el sitio web de BBVA Colombia, citando las fuentes.")

    if not st.session_state.messages:
        st.info(
            "Ejemplos: *¿Qué beneficios tiene la tarjeta Visa Aqua?* · *¿Qué es un CDT y a qué plazos?* · "
            "*¿Cómo abro una cuenta de ahorros digital?*"
        )

    def render_message(i: int, msg: dict) -> None:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Fuentes ({len(msg['sources'])})"):
                    for n, s in enumerate(msg["sources"], start=1):
                        st.markdown(f"[{n}] [{s['title']}]({s['url']}) · score {s['score']:.3f}")
            iid = msg.get("interaction_id")
            if iid and iid not in st.session_state.feedback_sent:
                c1, c2, _ = st.columns([1, 1, 8])
                for col, value, label in ((c1, 1, "👍"), (c2, -1, "👎")):
                    if col.button(label, key=f"fb{value}-{i}"):
                        api("POST", "/feedback", json={"interaction_id": iid, "value": value})
                        st.session_state.feedback_sent.add(iid)
                        st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        render_message(i, msg)

    if prompt := st.chat_input("Escribe tu pregunta sobre productos, servicios o trámites de BBVA Colombia"):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": [], "interaction_id": None})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Buscando en el sitio de BBVA…"):
                data, err = api("POST", "/chat", json={"message": prompt, "session_id": st.session_state.session_id})
        if err:
            st.session_state.messages.pop()
            st.error(err)
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": data["answer"], "sources": data["sources"],
                "interaction_id": data["interaction_id"],
            })
            st.rerun()

# ---------------- métricas ----------------
with tab_metrics:
    st.title("Analítica de conversaciones")
    m, err = api("GET", "/metrics")
    if err:
        st.error(err)
    elif not m["usage"]["total_questions"]:
        st.info("Aún no hay conversaciones registradas.")
    else:
        u, q, p, c, g, imp = (m[k] for k in ("usage", "quality", "performance", "content", "gaps", "impact"))
        pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
        k = st.columns(4)
        k[0].metric("Preguntas", u["total_questions"], help=f"{u['total_sessions']} sesiones")
        k[1].metric("Tasa de respuesta", pct(q["answer_rate"]))
        k[2].metric("Horas ahorradas (est.)", imp["estimated_hours_saved"],
                    help=f"{imp['minutes_saved_per_answer']} min por búsqueda manual evitada")
        k[3].metric("Latencia p50", f"{(p['latency_p50_ms'] or 0) / 1000:.1f} s",
                    help=f"p95: {(p['latency_p95_ms'] or 0) / 1000:.1f} s")
        k = st.columns(4)
        k[0].metric("Sesiones", u["total_sessions"])
        k[1].metric("Preguntas/sesión", u["avg_questions_per_session"])
        k[2].metric("Satisfacción", pct(q["satisfaction_rate"]), help=f"{q['feedback_count']} valoraciones")
        k[3].metric("Errores", pct(q["error_rate"]))

        left, right = st.columns(2)
        with left:
            st.subheader("Secciones del sitio más consultadas")
            bar_list(c["top_sections"])
            st.subheader("Preguntas por hora del día")
            hours = sorted(u["questions_by_hour"], key=int)
            bar_list([(f"{h}h", u["questions_by_hour"][h]) for h in hours], max_label_width=50)
        with right:
            st.subheader("Términos más frecuentes")
            bar_list(c["top_terms"])
            st.subheader("Brechas de contenido (sin respuesta)")
            if g["unanswered_questions"]:
                for qq in g["unanswered_questions"]:
                    st.markdown(f"- {qq}")
            else:
                st.caption("Todas las preguntas tuvieron respuesta.")
        with st.expander("Páginas más citadas"):
            simple_table(c["top_urls"], headers=("URL", "citas"))
