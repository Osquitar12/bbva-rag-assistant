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
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "span", "div"]):
        if el.name in ("div", "span") and el.find(["p", "div", "h1", "h2", "h3", "h4", "li", "span"]):
            continue  # evitar duplicar contenedores
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"{'#' * int(el.name[1])} {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


class ExtractMainContentStep(CleaningStep):
    """Usa trafilatura y, si se queda corto (páginas de producto con tarjetas),
    recurre a una extracción propia con BeautifulSoup."""

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
        ctx.text = extracted if len(extracted) >= 0.4 * len(fallback) else fallback
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
