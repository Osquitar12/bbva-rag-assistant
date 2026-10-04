"""Reporte de métricas: `python -m app.analytics.cli [--json]`."""
from __future__ import annotations

import argparse
import json

from app.analytics.metrics import ConversationAnalytics
from app.config import get_settings
from app.memory.repository import SQLAlchemyConversationRepository


def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def render(m: dict) -> str:
    u, q, p, c, g, i = (m[k] for k in ("usage", "quality", "performance", "content", "gaps", "impact"))
    lines = [
        "=" * 60, " REPORTE DE USO DEL ASISTENTE RAG - BBVA COLOMBIA", "=" * 60,
        "\n[USO]",
        f"  Sesiones: {u['total_sessions']}   Preguntas: {u['total_questions']}   "
        f"Preguntas/sesión: {u['avg_questions_per_session']}",
        f"  Preguntas de seguimiento: {_pct(u['follow_up_rate'])}   Hora pico: {u['peak_hour']}",
        "\n[CALIDAD]",
        f"  Tasa de respuesta: {_pct(q['answer_rate'])}   Sin respuesta: {_pct(q['no_answer_rate'])}   "
        f"Errores: {_pct(q['error_rate'])}",
        f"  Score medio de relevancia: {q['avg_top_score']}   Satisfacción: {_pct(q['satisfaction_rate'])} "
        f"({q['positive_feedback']}👍 / {q['negative_feedback']}👎)",
        "\n[RENDIMIENTO]",
        f"  Latencia p50: {p['latency_p50_ms']} ms   p95: {p['latency_p95_ms']} ms   "
        f"(recuperación p50: {p['retrieval_p50_ms']} ms, generación p50: {p['generation_p50_ms']} ms)",
        "\n[IMPACTO]",
        f"  Respuestas útiles: {i['answered_questions']}   Búsquedas manuales evitadas: {i['manual_searches_avoided']}",
        f"  Tiempo ahorrado estimado: {i['estimated_hours_saved']} h "
        f"(supuesto: {i['minutes_saved_per_answer']} min por búsqueda manual)",
        "\n[SECCIONES MÁS CONSULTADAS]",
        *[f"  {n:>4}  {s}" for s, n in c["top_sections"]],
        "\n[TÉRMINOS MÁS FRECUENTES]",
        "  " + ", ".join(f"{t} ({n})" for t, n in c["top_terms"]),
        "\n[BRECHAS DE CONTENIDO: preguntas sin respuesta]",
        *[f"  - {qq}" for qq in g["unanswered_questions"]],
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas del histórico de conversaciones")
    parser.add_argument("--json", action="store_true", help="salida JSON")
    args = parser.parse_args()
    s = get_settings()
    metrics = ConversationAnalytics(
        SQLAlchemyConversationRepository(s.resolved_database_url), s.minutes_saved_per_answer
    ).compute()
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) if args.json else render(metrics))


if __name__ == "__main__":
    main()
