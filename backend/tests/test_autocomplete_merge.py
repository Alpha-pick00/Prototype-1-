"""app.autocomplete.suggest_merged (로컬 SQLite 인덱스 기반 자동완성) 테스트."""

from __future__ import annotations

import asyncio

from app import autocomplete


def test_suggest_merged_returns_local_suggestions(monkeypatch):
    monkeypatch.setattr(autocomplete, "suggest", lambda prefix, limit=8: ["로컬상품1", "로컬상품2"])

    result = asyncio.run(autocomplete.suggest_merged("아무거나"))

    assert result == ["로컬상품1", "로컬상품2"]


def test_suggest_merged_respects_limit(monkeypatch):
    seen_limit: list[int] = []

    def _fake_suggest(prefix, limit=8):
        seen_limit.append(limit)
        return ["로컬A", "로컬B", "로컬C"][:limit]

    monkeypatch.setattr(autocomplete, "suggest", _fake_suggest)

    result = asyncio.run(autocomplete.suggest_merged("아무거나3", limit=2))

    assert result == ["로컬A", "로컬B"]
    assert seen_limit == [2]


def test_suggest_merged_returns_empty_for_blank_prefix(monkeypatch):
    monkeypatch.setattr(autocomplete, "suggest", lambda prefix, limit=8: ["안뜨면안됨"])

    assert asyncio.run(autocomplete.suggest_merged("   ")) == []
