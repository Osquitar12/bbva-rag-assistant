"""Descarga concurrente y educada (rate limit + reintentos) de páginas HTML.

Dos estrategias con la misma interfaz (`fetch_all`, `fetch_text`):
- `BrowserFetcher`: Chromium headless vía Playwright. Es la que usa el sitio de BBVA,
  cuyo WAF responde 403 a clientes HTTP que no son un navegador.
- `PageFetcher`: httpx puro, más ligero, para sitios sin esa restricción.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.exceptions import ScrapingError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = (429, 500, 502, 503, 504)


@dataclass
class FetchResult:
    url: str
    status: int
    html: str | None
    error: str | None = None


class _RetryableStatus(Exception):
    pass


class PageFetcher:
    def __init__(self, user_agent: str, concurrency: int = 4, delay: float = 0.5, timeout: float = 20.0):
        self._headers = {"User-Agent": user_agent, "Accept-Language": "es-CO,es;q=0.9"}
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._delay = delay
        self._timeout = timeout

    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatus)),
            reraise=True,
        ):
            with attempt:
                resp = await client.get(url)
                if resp.status_code in _RETRYABLE_STATUSES:
                    raise _RetryableStatus(f"HTTP {resp.status_code}")
                return resp
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _fetch_one(self, client: httpx.AsyncClient, url: str) -> FetchResult:
        async with self._semaphore:
            try:
                resp = await self._get(client, url)
                ctype = resp.headers.get("content-type", "")
                if resp.status_code != 200:
                    return FetchResult(url, resp.status_code, None, f"HTTP {resp.status_code}")
                if "html" not in ctype:
                    return FetchResult(url, resp.status_code, None, f"content-type no HTML: {ctype}")
                return FetchResult(url, resp.status_code, resp.text)
            except (httpx.HTTPError, _RetryableStatus) as exc:
                logger.warning("Fallo descargando %s: %s", url, exc)
                return FetchResult(url, 0, None, str(exc))
            finally:
                await asyncio.sleep(self._delay)

    async def fetch_all(self, urls: list[str]) -> list[FetchResult]:
        async with httpx.AsyncClient(
            headers=self._headers, timeout=self._timeout, follow_redirects=True
        ) as client:
            tasks = [self._fetch_one(client, u) for u in urls]
            results: list[FetchResult] = []
            for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
                results.append(await coro)
                if i % 25 == 0 or i == len(tasks):
                    logger.info("Descargadas %d/%d páginas", i, len(tasks))
            return results

    def fetch_text(self, url: str) -> str:
        """Cuerpo de un recurso no HTML (robots.txt, sitemap.xml)."""
        try:
            resp = httpx.get(url, headers=self._headers, timeout=self._timeout, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ScrapingError(str(exc)) from exc
        return resp.text


class BrowserFetcher:
    """Misma interfaz que `PageFetcher`, pero descargando con Chromium headless."""

    def __init__(self, user_agent: str, concurrency: int = 4, delay: float = 0.5, timeout: float = 20.0):
        self._user_agent = user_agent
        self._concurrency = max(1, concurrency)
        self._delay = delay
        self._timeout_ms = timeout * 1000

    async def _get(self, context, url: str) -> tuple[int, str, str]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, max=10),
            retry=retry_if_exception_type(_RetryableStatus),
            reraise=True,
        ):
            with attempt:
                page = await context.new_page()
                try:
                    resp = await page.goto(url, wait_until="commit", timeout=self._timeout_ms)
                    if resp is None:
                        raise _RetryableStatus("sin respuesta")
                    if resp.status in _RETRYABLE_STATUSES:
                        raise _RetryableStatus(f"HTTP {resp.status}")
                    # HTML tal como lo envía el servidor (igual que con httpx), no el DOM renderizado
                    return resp.status, resp.headers.get("content-type", ""), await resp.text()
                except _RetryableStatus:
                    raise
                except Exception as exc:  # timeouts y errores de red de Playwright
                    raise _RetryableStatus(str(exc).splitlines()[0]) from exc
                finally:
                    await page.close()
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _fetch_one(self, context, semaphore: asyncio.Semaphore, url: str, html_only: bool) -> FetchResult:
        async with semaphore:
            try:
                status, ctype, body = await self._get(context, url)
                if status != 200:
                    return FetchResult(url, status, None, f"HTTP {status}")
                if html_only and "html" not in ctype:
                    return FetchResult(url, status, None, f"content-type no HTML: {ctype}")
                return FetchResult(url, status, body)
            except _RetryableStatus as exc:
                logger.warning("Fallo descargando %s: %s", url, exc)
                return FetchResult(url, 0, None, str(exc))
            finally:
                await asyncio.sleep(self._delay)

    async def _fetch(self, urls: list[str], html_only: bool) -> list[FetchResult]:
        try:
            from playwright.async_api import async_playwright  # import perezoso: dependencia pesada
        except ImportError as exc:
            raise ScrapingError(
                "Falta Playwright. Instala con `pip install playwright && playwright install chromium` "
                "o usa SCRAPE_FETCHER=http"
            ) from exc

        semaphore = asyncio.Semaphore(self._concurrency)
        async with async_playwright() as p:
            # channel="chromium" = navegador completo en modo headless (no el headless shell)
            browser = await p.chromium.launch(
                headless=True, channel="chromium", args=["--disable-dev-shm-usage"]
            )
            try:
                context = await browser.new_context(user_agent=self._user_agent, locale="es-CO")
                tasks = [self._fetch_one(context, semaphore, u, html_only) for u in urls]
                results: list[FetchResult] = []
                for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
                    results.append(await coro)
                    if html_only and (i % 25 == 0 or i == len(tasks)):
                        logger.info("Descargadas %d/%d páginas", i, len(tasks))
                return results
            finally:
                await browser.close()

    async def fetch_all(self, urls: list[str]) -> list[FetchResult]:
        return await self._fetch(urls, html_only=True)

    def fetch_text(self, url: str) -> str:
        """Cuerpo de un recurso no HTML (robots.txt, sitemap.xml)."""
        result = asyncio.run(self._fetch([url], html_only=False))[0]
        if result.html is None:
            raise ScrapingError(result.error or "sin contenido")
        return result.html


def create_fetcher(settings: Settings) -> PageFetcher | BrowserFetcher:
    fetchers = {"browser": BrowserFetcher, "http": PageFetcher}
    try:
        cls = fetchers[settings.scrape_fetcher.lower()]
    except KeyError:
        raise ScrapingError(
            f"SCRAPE_FETCHER inválido: {settings.scrape_fetcher!r} (opciones: {', '.join(fetchers)})"
        ) from None
    return cls(
        settings.scrape_user_agent,
        settings.scrape_concurrency,
        settings.scrape_delay_seconds,
        settings.scrape_timeout_seconds,
    )
