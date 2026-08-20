import sqlite3
import time

from app import search_cache
from app.schemas import SearchResult

RESULT = [SearchResult(title="상품", url="https://coupang.com/vp/products/1", snippet="설명")]


def _backdate(db_path, query_key: str, created_at: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE search_cache SET created_at = ? WHERE query_key = ?", (created_at, query_key))
    conn.commit()
    conn.close()


def test_get_returns_none_when_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")

    assert search_cache.get("무선 이어폰") is None


def test_set_then_get_returns_cached_results(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")

    search_cache.set("무선 이어폰", RESULT)

    cached = search_cache.get("무선 이어폰")
    assert cached is not None
    assert cached[0].url == RESULT[0].url


def test_get_normalizes_whitespace_and_case(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")

    search_cache.set("무선  이어폰", RESULT)

    assert search_cache.get("무선 이어폰") is not None


def test_set_does_not_cache_empty_results(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")

    search_cache.set("무선 이어폰", [])

    assert search_cache.get("무선 이어폰") is None


def test_get_returns_none_when_expired(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")
    search_cache.set("무선 이어폰", RESULT)
    _backdate(tmp_path / "cache.db", "무선 이어폰", time.time() - search_cache.TTL_SECONDS - 1)

    assert search_cache.get("무선 이어폰") is None


def test_get_increments_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")
    search_cache.set("무선 이어폰", RESULT)

    search_cache.get("무선 이어폰")
    search_cache.get("무선 이어폰")

    conn = sqlite3.connect(tmp_path / "cache.db")
    hits = conn.execute("SELECT hits FROM search_cache WHERE query_key = ?", ("무선 이어폰",)).fetchone()[0]
    assert hits == 2


def test_set_overwrites_existing_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")
    search_cache.set("무선 이어폰", RESULT)

    updated = [SearchResult(title="새 상품", url="https://coupang.com/vp/products/2", snippet="설명2")]
    search_cache.set("무선 이어폰", updated)

    cached = search_cache.get("무선 이어폰")
    assert cached[0].url == updated[0].url


def test_top_queries_orders_by_hits_desc(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")
    search_cache.set("무선 이어폰", RESULT)
    search_cache.set("생수", RESULT)
    search_cache.get("생수")
    search_cache.get("생수")
    search_cache.get("무선 이어폰")

    assert search_cache.top_queries(limit=10) == ["생수", "무선 이어폰"]


def test_top_queries_excludes_below_min_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "DB_PATH", tmp_path / "cache.db")
    search_cache.set("무선 이어폰", RESULT)
    search_cache.set("생수", RESULT)
    search_cache.get("생수")

    assert search_cache.top_queries(limit=10, min_hits=1) == ["생수"]
