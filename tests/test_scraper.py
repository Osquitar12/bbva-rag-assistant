from pathlib import Path

from app.scraper.cleaner import HTMLCleaner, section_from_url
from app.scraper.sitemap import filter_entries, parse_sitemap
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


def test_cleaner_removes_boilerplate_and_keeps_content():
    doc = HTMLCleaner().clean(URL, FIXTURE.read_text())
    assert doc.title == "Tarjeta de Crédito Visa Aqua"
    assert "CVV es dinámico" in doc.text
    assert "## Beneficios" in doc.text
    for noise in ("cookies", "Copyright", "tracking", "Empresas"):
        assert noise not in doc.text
    assert doc.text.count("Solicítala ya") == 1
    assert doc.section == "personas/productos/tarjetas"


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
