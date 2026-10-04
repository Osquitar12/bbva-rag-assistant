from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import ScrapingError
from app.scraper.cleaner import HTMLCleaner, section_from_url
from app.scraper.fetcher import BrowserFetcher, PageFetcher, create_fetcher
from app.scraper.sitemap import (
    SitemapEntry,
    discover_urls,
    filter_entries,
    parse_sitemap,
    prioritize_entries,
)
from app.scraper.storage import LocalStorage, url_to_slug

FIXTURE = Path(__file__).parent / "fixtures" / "product_page.html"
URL = "https://www.bbva.com.co/personas/productos/tarjetas/credito/visa/aqua.html"


def test_parse_and_filter_sitemap():
    xml = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://a.co/personas/x.html</loc><lastmod>2026-01-01</lastmod></url>
      <url><loc>https://a.co/personas/x.html</loc></url>
      <url><loc>https://a.co/personas/investor-relations/y.html</loc></url>
      <url><loc>https://a.co/doc.pdf</loc></url></urlset>"""
    entries, children = parse_sitemap(xml)
    assert len(entries) == 4 and children == []
    kept = filter_entries(entries, [], ["/investor-relations/"], None, "ua")
    assert [e.url for e in kept] == ["https://a.co/personas/x.html"]
    assert kept[0].lastmod == "2026-01-01"


def test_discover_urls_with_injected_fetcher():
    pages = {
        "https://a.co/robots.txt": "User-agent: *\nDisallow: /personas/cards",
        "https://a.co/sitemap.xml": """<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>https://a.co/sm1.xml</loc></sitemap></sitemapindex>""",
        "https://a.co/sm1.xml": """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://a.co/personas/x.html</loc></url>
          <url><loc>https://a.co/personas/cards/y.html</loc></url></urlset>""",
    }
    found = discover_urls(
        "https://a.co/sitemap.xml", "https://a.co", "ua", [], [], fetch_text=pages.__getitem__
    )
    assert [e.url for e in found] == ["https://a.co/personas/x.html"]


def test_prioritize_entries_keeps_sitemap_order_within_groups():
    urls = ["https://a.co/blog/1.html", "https://a.co/productos/a.html",
            "https://a.co/blog/2.html", "https://a.co/productos/b.html"]
    ordered = prioritize_entries([SitemapEntry(u) for u in urls], ["/productos/"])
    assert [e.url for e in ordered] == [urls[1], urls[3], urls[0], urls[2]]


def test_create_fetcher_selects_strategy():
    assert isinstance(create_fetcher(Settings(scrape_fetcher="browser")), BrowserFetcher)
    assert isinstance(create_fetcher(Settings(scrape_fetcher="http")), PageFetcher)
    with pytest.raises(ScrapingError):
        create_fetcher(Settings(scrape_fetcher="otro"))


def test_cleaner_removes_boilerplate_and_keeps_content():
    doc = HTMLCleaner().clean(URL, FIXTURE.read_text())
    assert doc.title == "Tarjeta de Crédito Visa Aqua"
    assert "CVV es dinámico" in doc.text
    assert "## Beneficios" in doc.text
    for noise in ("cookies", "Copyright", "tracking", "Empresas"):
        assert noise not in doc.text
    assert doc.text.count("Solicítala ya") == 1
    assert doc.section == "personas/productos/tarjetas"


def test_cleaner_keeps_timeline_and_accordion_headings():
    paragraph = "BBVA acompaña a sus clientes con productos y servicios financieros en todo el país. " * 3
    html = f"""<html><head><title>Historia | BBVA Colombia</title></head><body><main>
      <h1>Historia de BBVA en Colombia</h1><p>{paragraph}</p>
      <div class="infographics"><h3>1996</h3><div class="rte">BBV adquiere acciones del Banco Ganadero.</div></div>
      <div class="infographics"><h3>2004</h3><div class="rte">La entidad pasa a llamarse BBVA Colombia.</div></div>
      <div class="accordion"><h5><button type="button"><span>Requisitos</span></button></h5>
        <div class="hidden"><p>Ser mayor de edad y presentar tu documento de identidad.</p></div></div>
      <button type="button">Solicitar ahora</button>
    </main></body></html>"""
    text = HTMLCleaner().clean("https://www.bbva.com.co/personas/historia.html", html).text
    assert "### 2004\nLa entidad pasa a llamarse BBVA Colombia." in text
    assert "##### Requisitos\nSer mayor de edad" in text
    assert "Solicitar ahora" not in text


def test_section_and_slug():
    assert section_from_url("https://x.co/") == "home"
    assert url_to_slug(URL) != url_to_slug(URL + "?a")


def test_storage_roundtrip(tmp_path):
    storage = LocalStorage(tmp_path / "raw", tmp_path / "clean")
    storage.save_raw(URL, "<html></html>", 200, None)
    doc = HTMLCleaner().clean(URL, FIXTURE.read_text())
    storage.save_clean(doc)
    assert storage.has_clean_data()
    assert storage.load_clean()[0].content_hash == doc.content_hash
    assert (tmp_path / "raw" / "index.jsonl").exists()
