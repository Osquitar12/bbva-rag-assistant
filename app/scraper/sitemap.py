"""Descubrimiento de URLs a partir del sitemap.xml (respetando robots.txt)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import httpx

from app.exceptions import ScrapingError

logger = logging.getLogger(__name__)
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    lastmod: str | None = None


def parse_sitemap(xml_text: str) -> tuple[list[SitemapEntry], list[str]]:
    """Devuelve (entradas de urlset, sub-sitemaps de un sitemapindex)."""
    try:
        root = ElementTree.fromstring(xml_text.strip().encode("utf-8"))
    except ElementTree.ParseError as exc:
        raise ScrapingError(f"Sitemap inválido: {exc}") from exc

    entries: list[SitemapEntry] = []
    for node in root.findall("sm:url", _NS):
        loc = node.findtext("sm:loc", default="", namespaces=_NS).strip()
        if loc:
            entries.append(SitemapEntry(loc, node.findtext("sm:lastmod", namespaces=_NS)))
    children = [
        n.findtext("sm:loc", default="", namespaces=_NS).strip()
        for n in root.findall("sm:sitemap", _NS)
    ]
    return entries, [c for c in children if c]


def filter_entries(
    entries: list[SitemapEntry],
    include: list[str],
    exclude: list[str],
    robots: RobotFileParser | None,
    user_agent: str,
) -> list[SitemapEntry]:
    seen: set[str] = set()
    result: list[SitemapEntry] = []
    for e in entries:
        url = e.url.split("#")[0]
        path = urlparse(url).path
        if url in seen:
            continue
        if include and not any(p in path for p in include):
            continue
        if any(p in path for p in exclude):
            continue
        if robots is not None and not robots.can_fetch(user_agent, url):
            continue
        # Solo páginas HTML (los sitemaps traen también formularios y PDFs)
        if not (path.endswith(".html") or path.endswith("/")):
            continue
        seen.add(url)
        result.append(SitemapEntry(url, e.lastmod))
    return result


def _load_robots(client: httpx.Client, base_url: str) -> RobotFileParser | None:
    rp = RobotFileParser()
    try:
        resp = client.get(f"{base_url.rstrip('/')}/robots.txt")
        resp.raise_for_status()
        rp.parse(resp.text.splitlines())
        return rp
    except httpx.HTTPError as exc:
        logger.warning("No se pudo leer robots.txt (%s); se continúa sin él", exc)
        return None


def discover_urls(
    sitemap_url: str,
    base_url: str,
    user_agent: str,
    include: list[str],
    exclude: list[str],
    timeout: float = 20.0,
) -> list[SitemapEntry]:
    headers = {"User-Agent": user_agent}
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        robots = _load_robots(client, base_url)
        pending, entries = [sitemap_url], []
        visited: set[str] = set()
        while pending:
            sm = pending.pop()
            if sm in visited:
                continue
            visited.add(sm)
            try:
                resp = client.get(sm)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ScrapingError(f"No se pudo descargar el sitemap {sm}: {exc}") from exc
            found, children = parse_sitemap(resp.text)
            entries.extend(found)
            pending.extend(children)
    filtered = filter_entries(entries, include, exclude, robots, user_agent)
    logger.info("Sitemap: %d URLs encontradas, %d tras filtros", len(entries), len(filtered))
    return filtered
