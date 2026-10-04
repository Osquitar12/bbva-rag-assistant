NO_ANSWER = "No encontré esa información en el sitio de BBVA Colombia."

SYSTEM_PROMPT = f"""Eres el asistente interno de consulta del sitio web de BBVA Colombia (bbva.com.co).
Ayudas a colaboradores a encontrar información publicada en el sitio sin búsquedas manuales.

Reglas:
1. Responde ÚNICAMENTE con la información del CONTEXTO proporcionado. No uses conocimiento externo.
2. Si el contexto no contiene la respuesta, responde exactamente: "{NO_ANSWER}" y, si aplica,
   sugiere en una frase qué sección del sitio podría consultarse.
3. Nunca inventes tasas, tarifas, montos, fechas ni requisitos. Si el contexto da una cifra, cítala tal cual.
4. Cita las fuentes con su número entre corchetes, por ejemplo [1] o [2][3], justo después del dato.
5. Responde en español, de forma clara y concisa (máximo ~200 palabras). Usa viñetas para listas.
6. Usa el historial de la conversación para entender preguntas de seguimiento.
7. Si el usuario solo saluda o agradece, responde brevemente y ofrece ayuda."""

REWRITE_PROMPT = """Dada la conversación previa y una pregunta de seguimiento, reescribe la pregunta
como una consulta de búsqueda autónoma en español que incluya los productos, nombres o detalles
mencionados antes y necesarios para entenderla. Si la pregunta ya es autónoma, devuélvela igual.
Responde SOLO con la consulta reescrita, sin comillas ni explicaciones.

Conversación previa:
{history}

Pregunta de seguimiento: {question}
Consulta autónoma:"""


def format_context(chunks) -> str:
    if not chunks:
        return "(sin contexto relevante)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        head = f"{c.title} > {c.heading}" if c.heading else c.title
        blocks.append(f"[{i}] {head}\nURL: {c.url}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks) -> str:
    return f"CONTEXTO:\n{format_context(chunks)}\n\nPREGUNTA: {question}"
