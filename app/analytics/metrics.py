"""Analítica del histórico de conversaciones: uso, calidad, rendimiento, contenido e impacto."""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.memory.repository import ConversationRepository, InteractionRecord
from app.scraper.cleaner import section_from_url

_STOPWORDS = set(
    """a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales cuando cuanto
    cuanta cuantos cuantas de del desde donde dos el ella ellas ellos en entre era es esa esas ese eso esos
    esta estas este esto estos fue ha hay hace hacer la las le les lo los mas me mi mis mucho muy no nos o
    otra otro para pero poco por porque puede puedo que quien se sea ser si sin sobre son su sus tambien
    tengo tiene tienen todo tu tus un una unas uno unos y ya yo quiero saber necesito cual cuales dame dime
    favor hola gracias bbva banco como hago puedo""".split()
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zñ0-9]+", _strip_accents(text.lower()))
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    q = statistics.quantiles(values, n=100, method="inclusive")
    return round(q[int(p) - 1], 1)


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


@dataclass
class ConversationAnalytics:
    repository: ConversationRepository
    minutes_saved_per_answer: float = 4.0
    top_k: int = 10

    def compute(self) -> dict:
        records: list[InteractionRecord] = list(self.repository.iter_interactions())
        total = len(records)
        ok = [r for r in records if not r.error]
        answered = [r for r in ok if r.answered]
        errors = [r for r in records if r.error]
        sessions = Counter(r.session_id for r in records)

        by_day = Counter(r.created_at.date().isoformat() for r in records if r.created_at)
        by_hour = Counter(r.created_at.hour for r in records if r.created_at)

        feedback = [r.feedback for r in records if r.feedback is not None]
        pos = sum(1 for f in feedback if f > 0)

        sections, urls = Counter(), Counter()
        for r in answered:
            for s in r.sources or []:
                urls[s.get("url", "")] += 1
                sections[section_from_url(s.get("url", ""))] += 1
        terms = Counter(t for r in records for t in set(tokenize(r.question)))

        scores = [r.top_score for r in answered if r.top_score is not None]
        lat = [r.latency_ms for r in ok]
        follow_ups = sum(1 for r in ok if r.rewritten_query)

        hours_saved = len(answered) * self.minutes_saved_per_answer / 60
        return {
            "usage": {
                "total_sessions": len(sessions),
                "total_questions": total,
                "avg_questions_per_session": round(total / len(sessions), 2) if sessions else 0,
                "follow_up_rate": _rate(follow_ups, len(ok)),
                "questions_by_day": dict(sorted(by_day.items())),
                "questions_by_hour": {h: by_hour.get(h, 0) for h in range(24)},
                "peak_hour": by_hour.most_common(1)[0][0] if by_hour else None,
            },
            "quality": {
                "answer_rate": _rate(len(answered), total),
                "no_answer_rate": _rate(len(ok) - len(answered), total),
                "error_rate": _rate(len(errors), total),
                "avg_top_score": round(statistics.mean(scores), 4) if scores else None,
                "feedback_count": len(feedback),
                "positive_feedback": pos,
                "negative_feedback": len(feedback) - pos,
                "satisfaction_rate": _rate(pos, len(feedback)) if feedback else None,
            },
            "performance": {
                "latency_p50_ms": _percentile(lat, 50),
                "latency_p95_ms": _percentile(lat, 95),
                "retrieval_p50_ms": _percentile([r.retrieval_ms for r in ok], 50),
                "generation_p50_ms": _percentile([r.generation_ms for r in ok], 50),
            },
            "content": {
                "top_sections": sections.most_common(self.top_k),
                "top_urls": urls.most_common(self.top_k),
                "top_terms": terms.most_common(self.top_k),
            },
            "gaps": {
                "unanswered_questions": [r.question for r in reversed(ok) if not r.answered][: self.top_k * 2],
                "recent_errors": [r.error for r in reversed(errors)][:5],
            },
            "impact": {
                "answered_questions": len(answered),
                "minutes_saved_per_answer": self.minutes_saved_per_answer,
                "estimated_hours_saved": round(hours_saved, 2),
                "manual_searches_avoided": len(answered),
            },
        }
