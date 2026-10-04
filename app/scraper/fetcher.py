"""Descarga concurrente y educada (rate limit + reintentos) de páginas HTML."""
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

logger = logging.getLogger(__name__)


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
                if resp.status_code in (429, 500, 502, 503, 504):
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
