# ROADMAP: Sistema RAG BBVA Colombia (para ejecutar con Claude Code)

> **Estado:** ✅ Fases 1–4 completadas y commiteadas (configuración, scraper, limpieza, chunking, embeddings, Qdrant, 7 tests en verde). **Continuar desde la Fase 5.**
>
> **Cómo usarlo:** abre Claude Code ahí y pégale el **Prompt inicial** (al final). Claude Code debe ejecutar **una fase a la vez**, verificarla y hacer **un commit por fase** con el mensaje indicado.

---

## 0. Contexto y reglas para Claude Code

- Prueba técnica de ML/AI Engineer: RAG en Python sobre https://www.bbva.com.co/. **Plazo: 1 día**, así que conviene hacerlo simple, robusto y bien documentado, sin sobre-ingeniería.
- **Idioma:** código en inglés, docstrings, README y mensajes de commit en **español**.
- **Un commit por fase**, con mensajes descriptivos tipo Conventional Commits (`feat(scraper): ...`). Nada de un solo commit gigante, porque se evalúa la progresión.
- Antes de cada commit: `pytest -q` en verde y verificar que el código importa.
- Todo parámetro va al `.env`; no se hardcodean valores.
- Herramientas **gratuitas/open source**. La máquina del usuario tiene **poco disco**, así que nada de PyTorch ni de LLM locales por defecto.

## 1. Datos ya verificados del sitio (no hace falta re-investigarlos)

- `robots.txt`: `Allow: /`, con `Disallow: *.content.html` y `Disallow: /personas/cards`. Hay que respetarlo con `urllib.robotparser`.
- `https://www.bbva.com.co/sitemap.xml` es un **urlset** con ~700 URLs y `<lastmod>`. Secciones: `/personas/blog/educacion-financiera/*` (blog), `/personas/productos/*`, `/empresas/productos/*`, `/personas/investor-relations/*` (duplicado en inglés de `atencion-al-inversionista`) y URLs `/herramientas/...` (formularios/simuladores sin texto).
- Exclusiones por defecto: `/investor-relations/,/herramientas/,/landing/formulario`. Además, solo URLs que terminen en `.html` o `/`.
- **Groq retiró `llama-3.3-70b-versatile`** (agosto 2026). Se usa `openai/gpt-oss-120b` (open-weights) para responder y `openai/gpt-oss-20b` para reformular.

## 2. Stack elegido (y por qué)

| Pieza | Elección | Justificación |
|---|---|---|
| Scraping | `httpx` (async) + `tenacity` + `trafilatura` + `BeautifulSoup/lxml` | El sitemap da URLs exactas, así que no hace falta crawler. Async con semáforo y delay para ser respetuoso. trafilatura extrae el contenido principal, y BS4 sirve de fallback en páginas de producto hechas con componentes |
| Embeddings | `fastembed` → `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384d, 220 MB) | Multilingüe (el contenido es en español), ONNX sin PyTorch, así que la imagen queda ligera |
| Vector DB | **Qdrant** self-hosted en docker-compose | Gratis, de producción, con filtros por payload. En tests se usa `QdrantClient(":memory:")` |
| Reranker (bonus) | `fastembed` `TextCrossEncoder` → `jinaai/jina-reranker-v2-base-multilingual` (1.1 GB) | Único reranker multilingüe disponible en fastembed. Se desactiva con `RERANKER_ENABLED=false` si falta disco |
| LLM | **Groq** (tier gratis, API OpenAI-compatible) con **Ollama** como alternativa local | Sin descargas. Ambos se llaman por `httpx` al endpoint `/v1/chat/completions`, sin SDK extra |
| Historial | SQLite + SQLAlchemy 2.0 | Persistente y sin servicio extra. Se puede cambiar a Postgres con `DATABASE_URL` |
| API | FastAPI | Contrato claro, `/docs` automático |
| UI | Streamlit (pestañas Chat + Métricas) | UI funcional en pocas líneas |
| Config | `pydantic-settings` | `.env` tipado |

## 3. Estructura objetivo

```
bbva-rag/
├── app/
│   ├── config.py              # Settings (pydantic-settings) + get_settings() con lru_cache (Singleton)
│   ├── exceptions.py          # RAGError, ScrapingError, VectorStoreError, LLMError, LLMConfigurationError
│   ├── logging_config.py
│   ├── scraper/
│   │   ├── sitemap.py         # parse_sitemap, filter_entries (robots + include/exclude), discover_urls
│   │   ├── fetcher.py         # PageFetcher async: semáforo, delay, reintentos 429/5xx
│   │   ├── cleaner.py         # Chain of Responsibility de CleaningStep + HTMLCleaner
│   │   ├── storage.py         # LocalStorage: data/raw/*.html + index.jsonl, data/clean/*.json
│   │   └── run.py             # CLI python -m app.scraper.run
│   ├── ingestion/
│   │   ├── chunker.py         # StructuralChunker: por encabezados Markdown → recursivo por tamaño + overlap
│   │   └── run.py             # CLI indexación; --all = scraping + indexación
│   ├── rag/
│   │   ├── embeddings.py      # EmbeddingProvider (ABC) + FastEmbedProvider + create_embedder (Factory)
│   │   ├── vectorstore.py     # QdrantVectorStore + RetrievedChunk
│   │   ├── reranker.py        # Reranker (ABC): CrossEncoderReranker / NoOpReranker + factory
│   │   ├── llm.py             # LLMProvider (ABC): GroqLLM / OllamaLLM + LLMFactory
│   │   ├── prompts.py         # system prompt, prompt de reformulación
│   │   └── service.py         # RAGService (Facade)
│   ├── memory/
│   │   ├── models.py          # Session, Message, InteractionLog (SQLAlchemy)
│   │   └── repository.py      # ConversationRepository (ABC) + SQLAlchemyConversationRepository
│   ├── analytics/
│   │   ├── metrics.py         # ConversationAnalytics: recorre el histórico y calcula métricas
│   │   └── cli.py             # python -m app.analytics.cli [--json]
│   ├── api/main.py            # FastAPI
│   └── ui/streamlit_app.py
├── tests/  (fixtures/ con HTML de ejemplo)
├── data/raw/.gitkeep  data/clean/.gitkeep
├── Dockerfile  docker-compose.yml  .env.example  requirements.txt  README.md
```

## 4. Patrones de diseño (mínimo 3, se documentan 6)

1. **Strategy**: `LLMProvider` (Groq/Ollama), `EmbeddingProvider`, `Reranker` (CrossEncoder/NoOp). Permiten cambiar implementación sin tocar `RAGService`.
2. **Factory**: `LLMFactory.create(settings)`, `create_embedder`, `create_reranker`. Construyen la estrategia según el `.env`.
3. **Repository**: `ConversationRepository` aísla la persistencia del historial y las analíticas del dominio.
4. **Facade**: `RAGService.ask(session_id, question)` orquesta reformular → recuperar → rerankear → generar → persistir.
5. **Chain of Responsibility**: pasos de limpieza de HTML encadenados (`set_next`).
6. **Singleton**: `get_settings()` con `lru_cache`.

---

## 5. Fases (1 fase = 1 commit)

### ✅ Fase 1: Esqueleto y configuración
- `app/config.py` con **todos** los parámetros: scraping (`SCRAPE_MAX_PAGES=300`, `SCRAPE_CONCURRENCY=4`, `SCRAPE_DELAY_SECONDS=0.5`, include/exclude, `SCRAPE_FORCE`), chunking (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, `CHUNK_MIN_CHARS=80`), embeddings, Qdrant (`QDRANT_URL`, `QDRANT_COLLECTION`, `REINDEX`), recuperación (`RETRIEVAL_TOP_K=20`, `RERANK_TOP_N=4`, `RERANKER_ENABLED`, `MIN_RELEVANCE_SCORE`), LLM (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_REWRITE_MODEL`, `GROQ_API_KEY`, temperatura, max_tokens, timeout, reintentos), `HISTORY_WINDOW_N=6`, `QUERY_REWRITE_ENABLED=true`, `DATABASE_URL`, `MINUTES_SAVED_PER_ANSWER=4`, `MODEL_CACHE_DIR`.
- `exceptions.py`, `logging_config.py`, `requirements.txt`, `.gitignore` (ignora `data/raw/*`, `data/clean/*`, `*.db`, `.env`, `.cache/`), `.env.example` comentado.
- Commit: `chore: estructura inicial del proyecto y configuración externalizada`

### ✅ Fase 2: Descubrimiento y descarga
- `sitemap.py`: parsear urlset **y** sitemapindex, filtrar por robots, patrones y extensión, deduplicar.
- `fetcher.py`: `httpx.AsyncClient` con `User-Agent` identificable, `Accept-Language: es-CO`, `asyncio.Semaphore`, `sleep(delay)` y reintento exponencial ante 429/5xx/errores de red. Nunca debe lanzar excepción: devuelve `FetchResult(url, status, html|None, error)`.
- `storage.py`: slug estable (`path` + sha1 corto), `save_raw` + `index.jsonl` (url, file, status, lastmod, fetched_at).
- Commit: `feat(scraper): descubrimiento de URLs vía sitemap y descarga concurrente`

### ✅ Fase 3: Limpieza y datos limpios
- `CleaningStep` abstracto con `set_next/handle/process`. Pasos: `ExtractMetadataStep` (title, meta description) → `RemoveBoilerplateStep` → `ExtractMainContentStep` → `NormalizeWhitespaceStep` → `DeduplicateLinesStep`.
- **Cuidado (lección aprendida):** al quitar boilerplate por clase/id, comparar **cada clase como token** (`(^|[-_])(cookies?|breadcrumbs?|navbar|footer|skip|megamenu)([-_]|$)`) y por `role` (`navigation`, `banner`, `contentinfo`). Un regex amplio tipo `header|menu` borra los "hero" de producto con el H1. Las etiquetas `nav/header/footer/script/style/form/svg/iframe/noscript/button` se eliminan directamente.
- Extracción: `trafilatura.extract(output_format="markdown", include_tables=True, favor_recall=True)`. Si el resultado mide menos del 40 % del fallback BS4 (que convierte h1–h4 a `#` y li a `- `), usar el fallback.
- `CleanDocument(url, title sin " | BBVA...", description, section = 3 primeros segmentos del path, text, lastmod, content_hash, scraped_at)` guardado como JSON.
- `run.py`: idempotente (si hay datos limpios y no `SCRAPE_FORCE`, omite), descarta textos menores a `CHUNK_MIN_CHARS` y duplicados por hash, y una página rota no tumba el proceso. Log final con limpias/fallidas/vacías.
- Test con `tests/fixtures/product_page.html` (header, nav, banner de cookies, main con h1/h2/li, footer). Debe conservar el contenido y eliminar cookies, footer y menú.
- Commit: `feat(scraper): pipeline de limpieza de HTML y almacenamiento de datos limpios`

### ✅ Fase 4: Chunking, embeddings e indexación
- `StructuralChunker`: separar por encabezados Markdown manteniendo la ruta ("Beneficios > Requisitos"), fusionar secciones muy cortas, aplicar split recursivo (`\n\n`, `\n`, `. `, ` `) hasta `CHUNK_SIZE` y overlap cortado en límite de palabra. ID determinista `uuid5(url#idx)` para que el upsert sea idempotente. Propiedad `embedding_text = "título > encabezado\ntexto"`.
- `FastEmbedProvider` (import perezoso, `cache_dir` en volumen, `query_embed` para consultas).
- `QdrantVectorStore`: `ensure_collection` (COSINE), `recreate`, `upsert` por lotes con payload (text, url, title, heading, section, lastmod), `search` con `query_points`, `count`.
- `ingestion/run.py`: `wait_for_qdrant`, omitir si ya hay vectores (salvo `REINDEX`), y `--all` = scraping + indexación.
- Tests: chunker (tamaños, overlap, encabezados) y vectorstore con `QdrantClient(":memory:")` + un `FakeEmbedder` determinista.
- Commit: `feat(ingestion): chunking estructural, embeddings multilingües e indexación en Qdrant`

### Fase 5: Proveedores LLM (Strategy + Factory)
- `LLMProvider.generate(messages, model=None, temperature, max_tokens) -> str`. La base `OpenAICompatibleLLM` hace POST con httpx a `{base_url}/chat/completions`, con reintentos ante 429/5xx/timeout y `LLMError` con mensaje claro.
- `GroqLLM` (si falta `GROQ_API_KEY`, lanza `LLMConfigurationError` con instrucciones) y `OllamaLLM`.
- `LLMFactory.create(settings)` con registro `{"groq": GroqLLM, "ollama": OllamaLLM}`.
- Tests con `httpx.MockTransport`.
- Commit: `feat(rag): proveedores de LLM intercambiables (Groq/Ollama) con patrón Strategy y Factory`

### Fase 6: Reranker (bonus)
- `CrossEncoderReranker` (fastembed `TextCrossEncoder.rerank(query, docs)`), que reemplaza `score` y conserva `retrieval_score`. `NoOpReranker` recorta por similitud. `create_reranker(settings)`.
- Si el modelo no carga (sin disco o red), se registra un warning y se usa `NoOpReranker`, sin romper.
- Commit: `feat(rag): reranker cross-encoder multilingüe con degradación elegante`

### Fase 7: Historial persistente (Repository)
- Modelos: `Session(id, created_at, updated_at)`, `Message(id, session_id, role, content, created_at)`, `InteractionLog(id, session_id, message_id, question, rewritten_query, answer, answered: bool, latency_ms, retrieval_ms, generation_ms, top_score, sources_json, num_chunks, error, feedback: int|null, created_at)`.
- `ConversationRepository`: `get_or_create_session`, `add_message`, `get_last_messages(session_id, n)` (los **N últimos** en orden cronológico), `list_sessions`, `get_history`, `log_interaction`, `set_feedback`, `iter_interactions`.
- Tests con SQLite en memoria; verificar que la ventana N se respeta.
- Commit: `feat(memory): historial de conversación persistente por session_id con patrón Repository`

### Fase 8: RAGService (Facade)
Flujo de `ask(session_id, question)`:
1. Cargar los últimos `HISTORY_WINDOW_N` mensajes.
2. Si hay historial y `QUERY_REWRITE_ENABLED`, **reformular** la pregunta en una consulta autónoma con `LLM_REWRITE_MODEL` (por ejemplo, "¿y cuál es la cuota de manejo?" pasa a "cuota de manejo tarjeta Visa Aqua BBVA"). Si falla, usar la original.
3. Embedding de la consulta → Qdrant top-K → reranker top-N → filtrar por `MIN_RELEVANCE_SCORE`.
4. Prompt: sistema en español ("responde solo con el contexto; si no está, di exactamente `No encontré esa información en el sitio de BBVA Colombia.`; cita fuentes [1], [2]; no inventes tasas ni cifras"), luego historial y después contexto numerado con título/URL y la pregunta.
5. Generar, detectar `answered` (no contiene la frase de "no encontré"), persistir mensajes + `InteractionLog` con tiempos por etapa y fuentes.
6. Devolver `{answer, sources[{title,url,score}], session_id, rewritten_query, latency_ms}`.
- Errores del LLM o de Qdrant: guardar el log con `error` y devolver un mensaje amable (la API responde 503 con detalle).
- Test end-to-end con fakes (FakeEmbedder, FakeLLM, Qdrant `:memory:`, SQLite memoria), incluyendo un segundo turno que use la reformulación.
- Commit: `feat(rag): servicio RAG como Facade con reformulación de consultas por historial`

### Fase 9: API FastAPI
- `POST /chat {session_id?, message}` (genera UUID si no viene), `GET /sessions`, `GET /sessions/{id}/history`, `POST /feedback {interaction_id, value: 1|-1}`, `GET /metrics`, `GET /health` (Qdrant + nº de vectores + proveedor LLM).
- Dependencias construidas una vez en `lifespan`. Handlers globales para `RAGError` → 503 y validación → 422.
- Tests con `TestClient` y el servicio inyectado con fakes.
- Commit: `feat(api): API REST con FastAPI para chat, sesiones, feedback y métricas`

### Fase 10: UI Streamlit
- Sidebar: ID de sesión (nuevo / elegir existente / pegar uno), botón "Nueva conversación", estado de `/health`.
- Pestaña **Chat**: `st.chat_message`, fuentes en un expander con links y botones 👍/👎.
- Pestaña **Métricas**: KPIs y gráficos nativos de Streamlit consumiendo `/metrics`.
- Commit: `feat(ui): interfaz conversacional en Streamlit con selector de sesión y panel de métricas`

### Fase 11: Analítica de conversaciones (requisito obligatorio)
`ConversationAnalytics` recorre `InteractionLog` + `Message` y devuelve un dict con:
- **Uso:** total de sesiones, preguntas, promedio de turnos por sesión, preguntas por día y por hora (picos).
- **Calidad:** tasa de respuesta (`answered`), tasa de "no encontré", tasa de error, top score promedio, % de feedback positivo.
- **Rendimiento:** latencia p50/p95 total y por etapa (recuperación vs generación).
- **Contenido:** top secciones del sitio citadas (por ejemplo `personas/productos/tarjetas`), top URLs citadas y top términos de las preguntas (tokenización + stopwords en español).
- **Brechas:** lista de preguntas sin respuesta (oportunidades de contenido).
- **Impacto:** `horas_ahorradas = respuestas_exitosas × MINUTES_SAVED_PER_ANSWER / 60` y búsquedas manuales evitadas.
- CLI: `python -m app.analytics.cli` (tabla legible) y `--json`. Endpoint `/metrics` y pestaña en la UI.
- Script `scripts/seed_demo.py` opcional que hace 8–10 preguntas reales vía API para poblar métricas.
- Tests con datos sintéticos.
- Commit: `feat(analytics): métricas de uso, calidad, rendimiento e impacto sobre el histórico`

### Fase 12: Docker
- **Un solo Dockerfile** (python:3.12-slim, `pip install --no-cache-dir`, usuario no root) para todos los servicios.
- `docker-compose.yml`:
  - `qdrant` (`qdrant/qdrant`, volumen `qdrant_data`, healthcheck).
  - `ingest`: `python -m app.ingestion.run --all`, monta `./data:/app/data` (así los datos crudos y limpios quedan **en local**) y el volumen `models_cache`. Es idempotente.
  - `api`: `depends_on: ingest: condition: service_completed_successfully`, `qdrant: service_healthy`, puerto 8000 y healthcheck `/health`.
  - `ui`: Streamlit en 8501, `API_URL=http://api:8000`.
  - `ollama` bajo `profiles: ["ollama"]` (opcional, no arranca por defecto).
- Todo con `env_file: .env`. Comando único: `docker compose up --build`.
- `.dockerignore`.
- Commit: `chore(docker): Dockerfile y docker-compose para levantar todo con un solo comando`

### Fase 13: README y cierre
README en español con: descripción, diagrama de arquitectura (Mermaid), requisitos previos (Docker, API key de Groq gratis, espacio en disco aproximado), pasos (clonar → `cp .env.example .env` → poner key → `docker compose up --build` → abrir http://localhost:8501), primer arranque (scraping de 5–10 min), uso de la UI y de la API (ejemplos curl), tabla de variables de entorno, **patrones de diseño (cuál, dónde y por qué)**, stack con justificación, cómo correr tests y la analítica, **supuestos asumidos**, **limitaciones conocidas** (contenido renderizado por JS no capturado, simuladores sin texto, PDFs no procesados, tasas pueden quedar desactualizadas, límite de páginas, rate limit del tier gratis de Groq, detección de "no respuesta" basada en frase fija), y **futuras mejoras** (búsqueda híbrida BM25 + densa, re-scraping incremental por `lastmod`, evaluación con RAGAS, PDFs, streaming, auth, Postgres, observabilidad con Langfuse, caché semántica).
- Commit: `docs: README completo con instalación, arquitectura, patrones y limitaciones`

### Fase 14: Prueba real local (el usuario la corre)
1. `docker compose up --build` y revisar en logs que el scraping y la indexación terminaron.
2. Preguntas de prueba: "¿Qué beneficios tiene la tarjeta Visa Aqua?", luego "¿y qué requisitos pide?" (prueba de memoria), "¿Qué es un CDT y cuánto paga?", "¿Cómo abro una cuenta de ahorros digital?" y "¿Quién ganó el mundial?" (debe responder "no encontré").
3. Revisar la pestaña Métricas y `docker compose run --rm api python -m app.analytics.cli`.
4. Ajustar `.env` si hace falta y hacer un commit final de fixes: `fix: ajustes tras prueba end-to-end`.

---

## 6. Checklist final contra la rúbrica

- [ ] Scraping de bbva.com.co → `data/raw` (HTML) y `data/clean` (JSON)
- [ ] Vectorización e indexación en Qdrant
- [ ] Interfaz conversacional (Streamlit + API)
- [ ] Historial por ID con N configurable y persistente (SQLite)
- [ ] `docker compose up --build` levanta todo
- [ ] Repo público con ≥13 commits descriptivos y progresivos
- [ ] ≥3 patrones documentados en el README
- [ ] Analítica de conversaciones (CLI + endpoint + UI)
- [ ] README con todas las secciones pedidas
- [ ] Bonus: reranker, manejo de errores, `.env`

---

## Prompt inicial para pegar en Claude Code

```
Lee ROADMAP.md completo. Vas a construir este proyecto fase por fase.
Reglas:
- Ejecuta UNA fase a la vez, en orden.
- Al terminar cada fase: corre `pytest -q`, verifica imports y haz `git commit`
  con el mensaje indicado en la fase (cuerpo con viñetas de lo que hiciste).
- No avances a la siguiente fase si los tests fallan.
- Si una decisión no está en el roadmap, toma la más simple, anótala para la
  sección "Supuestos" del README y sigue.
- Al final de cada fase dime en 2 líneas qué quedó hecho.
Las fases 1 a 4 ya están hechas: revisa el código existente en app/ y tests/
para seguir sus convenciones. Empieza por la Fase 5.
```

Haz `git push` al terminar cada fase (el remoto ya es https://github.com/Osquitar12/bbva-rag-assistant), y el repo debe ser **público**.
