"""app.category.classify_category(Groq 기반 카테고리 자동 분류) 테스트.
네트워크 요청 금지 - 전부 monkeypatch."""

from __future__ import annotations

import asyncio

from app import category


def _fake_client(content: str):
    class _FakeMessage:
        pass

    message = _FakeMessage()
    message.content = content

    class _FakeChoice:
        pass

    choice = _FakeChoice()
    choice.message = message

    class _FakeResponse:
        choices = [choice]

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    return _FakeClient()


def test_classify_category_returns_none_when_category_list_empty(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", "fake-key")

    async def _boom():
        raise AssertionError("후보가 없는데 Groq를 불렀다")

    monkeypatch.setattr(category, "_client", _boom)

    assert asyncio.run(category.classify_category("초코파이", [])) is None


def test_classify_category_returns_none_when_key_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", None)

    async def _boom():
        raise AssertionError("키가 없는데 Groq를 불렀다")

    monkeypatch.setattr(category, "_client", _boom)

    assert asyncio.run(category.classify_category("초코파이", ["과자/간식"])) is None


def test_classify_category_returns_picked_name_on_success(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", "fake-key")
    monkeypatch.setattr(category, "_client", lambda: _fake_client('{"index": 1}'))

    result = asyncio.run(category.classify_category("초코파이", ["수산", "과자/간식", "주방용품"]))

    assert result == "과자/간식"


def test_classify_category_returns_none_for_out_of_range_index(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", "fake-key")
    monkeypatch.setattr(category, "_client", lambda: _fake_client('{"index": 9}'))

    assert asyncio.run(category.classify_category("초코파이", ["과자/간식"])) is None


def test_classify_category_returns_none_on_malformed_response(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", "fake-key")
    monkeypatch.setattr(category, "_client", lambda: _fake_client("이건 JSON이 아님"))

    assert asyncio.run(category.classify_category("초코파이", ["과자/간식"])) is None
