"""Chunking estructural: primero por encabezados Markdown, luego recursivo por tamaño."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.scraper.storage import CleanDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str  # texto que se muestra al LLM
    url: str
    title: str
    section: str
    heading: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """Texto enriquecido con contexto para mejorar el embedding."""
        head = f"{self.title} > {self.heading}" if self.heading else self.title
        return f"{head}\n{self.text}"


def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
    """Devuelve (ruta de encabezados, cuerpo) por cada sección del documento."""
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(stack), body))
        buf.clear()

    for line in text.split("\n"):
        m = _HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


def _merge_sections(
    prev: tuple[list[str], str], cur: tuple[list[str], str]
) -> tuple[list[str], str]:
    """Une dos secciones bajo su ruta común, dejando en el texto el encabezado propio
    de cada una. Así en una línea de tiempo cada año acompaña a su hito dentro del chunk."""
    (prev_path, prev_body), (cur_path, cur_body) = prev, cur
    common: list[str] = []
    for a, b in zip(prev_path, cur_path):
        if a != b:
            break
        common.append(a)

    def render(path: list[str], body: str) -> str:
        own = " > ".join(path[len(common):])
        return f"{own}\n{body}" if own else body

    return common, f"{render(prev_path, prev_body)}\n{render(cur_path, cur_body)}"


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) > size:
            pieces.extend(_recursive_split(part, size, rest))
            current = ""
        else:
            current = part
    if current:
        pieces.append(current)
    return pieces


def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    result = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else tail
        result.append(f"{tail} {cur}".strip())
    return result


def drop_repeated_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Deja una sola copia de los chunks cuyo texto se repite entre páginas.

    El sitio repite bloques enteros (preguntas frecuentes, "También te puede
    interesar") en cientos de páginas; indexados tal cual, copan los resultados de
    la búsqueda y dejan fuera la página que sí responde. Se conserva la copia de la
    URL más corta, que suele ser la página más general.
    """
    best: dict[str, Chunk] = {}
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.text).strip().lower()
        kept = best.get(key)
        if kept is None or len(chunk.url) < len(kept.url):
            best[key] = chunk
    keep = {id(c) for c in best.values()}
    return [c for c in chunks if id(c) in keep]


class StructuralChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, min_chars: int = 80):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        self.chunk_size, self.chunk_overlap, self.min_chars = chunk_size, chunk_overlap, min_chars

    def split(self, doc: CleanDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        sections = _split_by_headings(doc.text) or [([], doc.text)]
        # Fusiona secciones muy pequeñas con la siguiente para no perder contexto
        merged: list[tuple[list[str], str]] = []
        for section in sections:
            if merged and len(merged[-1][1]) < self.min_chars * 2:
                merged.append(_merge_sections(merged.pop(), section))
            else:
                merged.append(section)

        for path, body in merged:
            heading = " > ".join(path)
            pieces = _with_overlap(_recursive_split(body, self.chunk_size, _SEPARATORS), self.chunk_overlap)
            for piece in pieces:
                piece = piece.strip()
                if len(piece) < self.min_chars and chunks:
                    continue
                idx = len(chunks)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.url}#{idx}")),
                        text=piece,
                        url=doc.url,
                        title=doc.title,
                        section=doc.section,
                        heading=heading,
                        chunk_index=idx,
                        metadata={"lastmod": doc.lastmod, "content_hash": doc.content_hash},
                    )
                )
        return chunks
