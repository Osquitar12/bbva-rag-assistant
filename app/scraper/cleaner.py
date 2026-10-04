"""Limpieza de HTML como una cadena de pasos (Chain of Responsibility / Pipeline).

Cada `CleaningStep` recibe un `CleaningContext`, lo transforma y lo pasa al
siguiente. Añadir, quitar o reordenar pasos no exige tocar los demás.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup

from app.scraper.storage import CleanDocument

_BOILERPLATE_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "nav", "header", "footer", "button"]
# Se compara contra cada clase/id por separado y como "token" para no borrar
# por accidente componentes de contenido (p. ej. "hero-header" con el título).
_BOILERPLATE_HINTS = re.compile(r"(^|[-_])(cookies?|breadcrumbs?|navbar|footer|skip(-link)?|megamenu)([-_]|$)", re.I)
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
# Si trafilatura conserva menos de esta fracción de los encabezados de sección, se usa el fallback
_MIN_HEADINGS_KEPT = 0.8


@dataclass
class CleaningContext:
    url: str
    html: str
    lastmod: str | None = None
    soup: BeautifulSoup | None = None
    title: str = ""
    description: str = ""
    text: str = ""
    meta: dict = field(default_factory=dict)


class CleaningStep(ABC):
    def __init__(self) -> None:
        self._next: CleaningStep | None = None

    def set_next(self, step: "CleaningStep") -> "CleaningStep":
        self._next = step
        return step

    def handle(self, ctx: CleaningContext) -> CleaningContext:
        ctx = self.process(ctx)
        return self._next.handle(ctx) if self._next else ctx

    @abstractmethod
    def process(self, ctx: CleaningContext) -> CleaningContext: ...


class ExtractMetadataStep(CleaningStep):
    def process(self, ctx: CleaningContext) -> CleaningContext:
        ctx.soup = BeautifulSoup(ctx.html, "lxml")
        title_tag = ctx.soup.find("title")
        h1 = ctx.soup.find("h1")
        ctx.title = (title_tag.get_text(" ", strip=True) if title_tag else "") or (
            h1.get_text(" ", strip=True) if h1 else ""
        )
        desc = ctx.soup.find("meta", attrs={"name": "description"})
        ctx.description = (desc.get("content") or "").strip() if desc else ""
        return ctx


class RemoveBoilerplateStep(CleaningStep):
    """Elimina menús, footers, scripts, banners de cookies, etc."""

    def process(self, ctx: CleaningContext) -> CleaningContext:
        soup = ctx.soup
        assert soup is not None
        # Los títulos de los acordeones ("Requisitos", "Tarifas") son un <button> dentro
        # de un encabezado: se conserva su texto en lugar de borrarlo con el resto de botones
        for heading in soup(_HEADING_TAGS):
            for button in heading("button"):
                button.unwrap()
        for tag in soup(_BOILERPLATE_TAGS):
            tag.decompose()
        for tag in soup.find_all(True):
            if tag.attrs is None:
                continue
            tokens = [*tag.get("class", []), tag.get("id", "") or ""]
            role = tag.get("role", "") or ""
            if tag.name in ("body", "main", "html"):
                continue
            if role in ("navigation", "banner", "contentinfo") or any(
                t and _BOILERPLATE_HINTS.search(t) for t in tokens
            ):
                tag.decompose()
        return ctx


def _soup_to_markdown(soup: BeautifulSoup) -> str:
    """Fallback: texto estructurado conservando encabezados como Markdown."""
    root = soup.find("main") or soup.body or soup
    lines: list[str] = []
    for el in root.find_all([*_HEADING_TAGS, "p", "li", "td", "th", "span", "div"]):
        if el.name in ("div", "span") and (
            el.find([*_HEADING_TAGS, "p", "div", "li", "span"]) or el.find_parent(_HEADING_TAGS)
        ):
            continue  # evitar duplicar contenedores y el texto de los encabezados
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in _HEADING_TAGS:
            lines.append(f"{'#' * int(el.name[1])} {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def _keeps_section_headings(extracted: str, fallback: str) -> bool:
    """¿El texto de trafilatura conserva los encabezados de sección (h2-h6) del fallback?

    En páginas hechas con componentes (líneas de tiempo, acordeones) trafilatura
    suele quedarse con los párrafos y descartar los títulos, que son justo lo que
    da contexto a cada chunk ("2004", "Requisitos").
    """
    headings = {
        re.sub(r"\s+", " ", line.lstrip("# ")).lower()
        for line in fallback.splitlines()
        if line.startswith("##")
    }
    if len(headings) < 2:
        return True
    text = re.sub(r"\s+", " ", extracted).lower()
    return sum(h in text for h in headings) / len(headings) >= _MIN_HEADINGS_KEPT


class ExtractMainContentStep(CleaningStep):
    """Usa trafilatura y, si se queda corto o pierde los encabezados (páginas de
    producto con tarjetas), recurre a una extracción propia con BeautifulSoup."""

    def process(self, ctx: CleaningContext) -> CleaningContext:
        assert ctx.soup is not None
        fallback = _soup_to_markdown(ctx.soup)
        extracted = trafilatura.extract(
            ctx.html,
            output_format="markdown",
            include_tables=True,
            include_links=False,
            include_comments=False,
            favor_recall=True,
        ) or ""
        use_extracted = len(extracted) >= 0.4 * len(fallback) and _keeps_section_headings(extracted, fallback)
        ctx.text = extracted if use_extracted else fallback
        ctx.meta["extractor"] = "trafilatura" if ctx.text is extracted else "bs4"
        return ctx


class NormalizeWhitespaceStep(CleaningStep):
    def process(self, ctx: CleaningContext) -> CleaningContext:
        text = ctx.text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        ctx.text = text.strip()
        return ctx


class DeduplicateLinesStep(CleaningStep):
    """Quita líneas repetidas (CTAs, avisos legales repetidos) y líneas triviales."""

    def __init__(self, min_len: int = 3):
        super().__init__()
        self.min_len = min_len

    def process(self, ctx: CleaningContext) -> CleaningContext:
        seen: set[str] = set()
        out: list[str] = []
        for line in ctx.text.split("\n"):
            key = line.strip().lower()
            if key and (len(key) < self.min_len or key in seen):
                continue
            if key:
                seen.add(key)
            out.append(line)
        ctx.text = "\n".join(out).strip()
        return ctx


def section_from_url(url: str) -> str:
    """'/personas/productos/tarjetas/credito.html' -> 'personas/productos/tarjetas'."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    parts = parts[:-1] if parts and parts[-1].endswith(".html") else parts
    return "/".join(parts[:3]) or "home"


def build_default_pipeline() -> CleaningStep:
    head = ExtractMetadataStep()
    head.set_next(RemoveBoilerplateStep()).set_next(ExtractMainContentStep()).set_next(
        NormalizeWhitespaceStep()
    ).set_next(DeduplicateLinesStep())
    return head


class HTMLCleaner:
    def __init__(self, pipeline: CleaningStep | None = None):
        self.pipeline = pipeline or build_default_pipeline()

    def clean(self, url: str, html: str, lastmod: str | None = None) -> CleanDocument:
        ctx = self.pipeline.handle(CleaningContext(url=url, html=html, lastmod=lastmod))
        title = re.sub(r"\s*\|\s*BBVA.*$", "", ctx.title, flags=re.I).strip()
        return CleanDocument(
            url=url,
            title=title,
            description=ctx.description,
            section=section_from_url(url),
            text=ctx.text,
            lastmod=lastmod,
        )
