"""app/agents/groq.py::refine_query 테스트 - 네트워크 호출 없이 _client()만
가짜로 갈아끼운다(test_clarify_facets.py의 deepseek._client 목킹과 동일 패턴)."""

from __future__ import annotations

import asyncio

import pytest

from app.agents import groq


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self._content = content
        self._exc = exc

    async def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._content or "")


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def test_refine_query_returns_cleaned_text_from_model(monkeypatch):
    monkeypatch.setattr(groq, "_client", lambda: _FakeClient(_FakeCompletions('{"query": "충전기"}')))

    result = asyncio.run(groq.refine_query("안녕 나 충전기 사고싶어"))

    assert result == "충전기"


def test_refine_query_returns_original_on_api_exception(monkeypatch):
    monkeypatch.setattr(groq, "_client", lambda: _FakeClient(_FakeCompletions(exc=RuntimeError("API 오류"))))

    result = asyncio.run(groq.refine_query("안녕 나 충전기 사고싶어"))

    assert result == "안녕 나 충전기 사고싶어"


def test_refine_query_returns_original_when_response_missing_query_field(monkeypatch):
    monkeypatch.setattr(groq, "_client", lambda: _FakeClient(_FakeCompletions('{"unexpected": true}')))

    result = asyncio.run(groq.refine_query("충전기 살래"))

    assert result == "충전기 살래"


def test_refine_query_returns_original_when_response_is_not_valid_json(monkeypatch):
    monkeypatch.setattr(groq, "_client", lambda: _FakeClient(_FakeCompletions("이건 JSON이 아님")))

    result = asyncio.run(groq.refine_query("충전기 살래"))

    assert result == "충전기 살래"
