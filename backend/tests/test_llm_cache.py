"""app.llm_cache 테스트 - SUPABASE_URL/SUPABASE_KEY 미설정이면 모든 함수가
안전하게 no-op(캐시 미스)로 동작해야 한다. 네트워크 요청 금지."""

from __future__ import annotations

import asyncio

from app import llm_cache


def _reset_client_cache(monkeypatch):
    monkeypatch.setattr(llm_cache, "_client", None)
    monkeypatch.setattr(llm_cache, "_client_checked", False)


def test_exact_get_returns_none_when_supabase_not_configured(monkeypatch):
    _reset_client_cache(monkeypatch)
    monkeypatch.setattr("app.config.settings.supabase_url", None)
    monkeypatch.setattr("app.config.settings.supabase_key", None)

    assert asyncio.run(llm_cache.exact_get("ns", "질의")) is None


def test_exact_set_is_noop_when_supabase_not_configured(monkeypatch):
    _reset_client_cache(monkeypatch)
    monkeypatch.setattr("app.config.settings.supabase_url", None)
    monkeypatch.setattr("app.config.settings.supabase_key", None)

    async def _boom(*args, **kwargs):
        raise AssertionError("Supabase 미설정인데 클라이언트를 만들려 했다")

    monkeypatch.setattr(llm_cache, "acreate_client", _boom)

    asyncio.run(llm_cache.exact_set("ns", "질의", {"facets": []}))


def test_semantic_get_returns_none_when_embedding_unavailable(monkeypatch):
    _reset_client_cache(monkeypatch)
    monkeypatch.setattr("app.config.settings.supabase_url", "https://example.supabase.co")
    monkeypatch.setattr("app.config.settings.supabase_key", "fake-key")

    class _FakeClient:
        pass

    async def _fake_create_client(url, key):
        return _FakeClient()

    monkeypatch.setattr(llm_cache, "acreate_client", _fake_create_client)

    async def _no_embedding(texts):
        return None

    monkeypatch.setattr(llm_cache.embeddings, "embed", _no_embedding)

    assert asyncio.run(llm_cache.semantic_get("ns", "질의")) is None


def test_exact_get_returns_cached_response_on_hit(monkeypatch):
    _reset_client_cache(monkeypatch)
    monkeypatch.setattr("app.config.settings.supabase_url", "https://example.supabase.co")
    monkeypatch.setattr("app.config.settings.supabase_key", "fake-key")

    class _FakeResult:
        data = [{"response": {"facets": [{"label": "브랜드", "options": ["삼성"]}]}}]

    class _FakeExecutable:
        def eq(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        async def execute(self):
            return _FakeResult()

    class _FakeTable:
        def select(self, *a, **k):
            return _FakeExecutable()

    class _FakeClient:
        def table(self, name):
            return _FakeTable()

    async def _fake_create_client(url, key):
        return _FakeClient()

    monkeypatch.setattr(llm_cache, "acreate_client", _fake_create_client)

    result = asyncio.run(llm_cache.exact_get("ns", "질의"))

    assert result == {"facets": [{"label": "브랜드", "options": ["삼성"]}]}
