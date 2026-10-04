"""Persistencia local de datos crudos (HTML) y limpios (JSON)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def url_to_slug(url: str) -> str:
    path = urlparse(url).path.strip("/") or "home"
    path = path.removesuffix(".html").replace("/", "__")
    digest = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{path[:150]}-{digest}"


@dataclass
class CleanDocument:
    url: str
    title: str
    description: str
    section: str
    text: str
    lastmod: str | None = None
    content_hash: str = ""
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()


class LocalStorage:
    def __init__(self, raw_dir: Path, clean_dir: Path):
        self.raw_dir, self.clean_dir = raw_dir, clean_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, url: str, html: str, status: int, lastmod: str | None) -> Path:
        slug = url_to_slug(url)
        path = self.raw_dir / f"{slug}.html"
        path.write_text(html, encoding="utf-8")
        meta = {
            "url": url,
            "file": path.name,
            "status": status,
            "lastmod": lastmod,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        with (self.raw_dir / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return path

    def save_clean(self, doc: CleanDocument) -> Path:
        path = self.clean_dir / f"{url_to_slug(doc.url)}.json"
        path.write_text(json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_clean(self) -> list[CleanDocument]:
        docs = []
        for p in sorted(self.clean_dir.glob("*.json")):
            docs.append(CleanDocument(**json.loads(p.read_text(encoding="utf-8"))))
        return docs

    def has_clean_data(self) -> bool:
        return any(self.clean_dir.glob("*.json"))
