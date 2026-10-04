"""CLI de scraping: `python -m app.scraper.run [--max-pages N] [--force]`."""
from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.logging_config import setup_logging
from app.scraper.cleaner import HTMLCleaner
from app.scraper.fetcher import PageFetcher
from app.scraper.sitemap import discover_urls
from app.scraper.storage import LocalStorage

logger = logging.getLogger("scraper")


def run_scraping(max_pages: int | None = None, force: bool | None = None) -> int:
    s = get_settings()
    storage = LocalStorage(s.raw_dir, s.clean_dir)
    force = s.scrape_force if force is None else force
    if storage.has_clean_data() and not force:
        logger.info("Ya existen datos limpios en %s; se omite el scraping (usa SCRAPE_FORCE=true)", s.clean_dir)
        return 0

    entries = discover_urls(
        s.scrape_sitemap_url, s.scrape_base_url, s.scrape_user_agent,
        s.include_patterns, s.exclude_patterns, s.scrape_timeout_seconds,
    )[: max_pages or s.scrape_max_pages]
    lastmods = {e.url: e.lastmod for e in entries}

    fetcher = PageFetcher(s.scrape_user_agent, s.scrape_concurrency, s.scrape_delay_seconds, s.scrape_timeout_seconds)
    results = asyncio.run(fetcher.fetch_all([e.url for e in entries]))

    cleaner, saved, failed, empty = HTMLCleaner(), 0, 0, 0
    seen_hashes: set[str] = set()
    for r in results:
        if r.html is None:
            failed += 1
            continue
        storage.save_raw(r.url, r.html, r.status, lastmods.get(r.url))
        try:
            doc = cleaner.clean(r.url, r.html, lastmods.get(r.url))
        except Exception as exc:  # una página rota no debe tumbar el proceso
            logger.warning("Error limpiando %s: %s", r.url, exc)
            failed += 1
            continue
        if len(doc.text) < s.chunk_min_chars or doc.content_hash in seen_hashes:
            empty += 1
            continue
        seen_hashes.add(doc.content_hash)
        storage.save_clean(doc)
        saved += 1
    logger.info("Scraping terminado: %d limpias, %d fallidas, %d vacías/duplicadas", saved, failed, empty)
    return saved


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Scraper del sitio de BBVA Colombia")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--force", action="store_true", default=None)
    args = parser.parse_args()
    run_scraping(args.max_pages, args.force)


if __name__ == "__main__":
    main()
