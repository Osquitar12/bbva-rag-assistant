"""Genera conversaciones de demostración contra la API para poblar las métricas.

Uso (con el sistema levantado):
    python scripts/seed_demo.py                     # API en http://localhost:8000
    docker compose exec api python scripts/seed_demo.py
"""
from __future__ import annotations

import os
import random
import sys

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

CONVERSATIONS = [
    ["¿Qué beneficios tiene la tarjeta de crédito Visa Aqua?", "¿Y qué requisitos piden para solicitarla?"],
    ["¿Qué es un CDT?", "¿Se puede abrir en línea?", "¿Cuál es el plazo mínimo?"],
    ["¿Cómo abro una cuenta de ahorros digital?", "¿Tiene cuota de manejo?"],
    ["¿Qué tipos de crédito de vivienda ofrece BBVA?", "¿Financian vivienda para colombianos en el exterior?"],
    ["¿Cómo funcionan las llaves de Bre-B?"],
    ["¿Qué hago si me roban el celular?"],
    ["¿Qué seguros ofrecen para el hogar?"],
    ["¿Quién ganó el último mundial de fútbol?"],  # fuera de dominio: debe responder "no encontré"
]


def main() -> int:
    with httpx.Client(base_url=API_URL, timeout=120) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"La API no responde en {API_URL}: {exc}")
            return 1
        for convo in CONVERSATIONS:
            session_id = None
            for question in convo:
                resp = client.post("/chat", json={"message": question, "session_id": session_id})
                if resp.status_code != 200:
                    print(f"  ✗ {question} -> {resp.status_code}: {resp.text[:120]}")
                    break
                data = resp.json()
                session_id = data["session_id"]
                mark = "✓" if data["answered"] else "∅"
                print(f"  {mark} {question}  ({data['latency_ms']:.0f} ms)")
                if data["answered"] and random.random() < 0.7:
                    client.post("/feedback", json={"interaction_id": data["interaction_id"], "value": 1})
    print("Listo. Revisa la pestaña Métricas o ejecuta: python -m app.analytics.cli")
    return 0


if __name__ == "__main__":
    sys.exit(main())
