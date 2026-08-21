"""app.category.classify_category 테스트. 네트워크 요청 금지 - 전부 monkeypatch."""

import asyncio

from app import category
from app.config import settings
from app.schemas import SearchResult


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(self, content: str, seen_kwargs: dict):
        self._content = content
        self._seen_kwargs = seen_kwargs
        self.chat = self

    @property
    def completions(self):
        return self

    async def create(self, **kwargs):
        self._seen_kwargs.update(kwargs)
        return _FakeResponse(self._content)


def _install_fake_client(monkeypatch, content: str) -> dict:
    seen_kwargs: dict = {}
    monkeypatch.setattr(
        category, "AsyncOpenAI", lambda **_: _FakeClient(content, seen_kwargs)
    )
    return seen_kwargs


def test_classify_category_uses_the_refine_model_setting(monkeypatch):
    """토큰 절약(2026-08-19) - classify_category는 검색마다(스타일 가이드 게이트
    때문에) 무조건 한 번씩 불려서, 다른 슬롯(propose·judge·style_guide)이 쓰는
    groq_model이 아니라 groq_refine_model을 명시적으로 써야 한다(config.py 주석
    참고) - 이 자리에서 원래 Groq gpt-oss-120b/20b처럼 서로 다른 모델로 분리해
    토큰 예산을 나누던 취지가 유지되는지 확인한다. 2026-08-21 HCX 전환 이후
    groq_model과 groq_refine_model이 우연히 같은 값(HCX-005, 현재 HCX가 제공하는
    유일한 채팅 모델)이 될 수 있어 "값이 다르다"는 더 이상 검증하지 않는다 -
    두 설정이 각자 옳게 참조되는지만 본다."""
    seen_kwargs = _install_fake_client(
        monkeypatch, '{"category": "패션의류/잡화", "is_beverage": false}'
    )

    asyncio.run(category.classify_category("나이키 에어포스1", []))

    assert seen_kwargs["model"] == settings.groq_refine_model


def test_classify_category_parses_valid_category(monkeypatch):
    _install_fake_client(monkeypatch, '{"category": "뷰티", "is_beverage": false}')

    result = asyncio.run(category.classify_category("립스틱", []))

    assert result.category == "뷰티"
    assert result.is_beverage is False


def test_classify_category_marks_beverage_only_for_food_category(monkeypatch):
    _install_fake_client(monkeypatch, '{"category": "식품", "is_beverage": true}')

    result = asyncio.run(category.classify_category("생수 2L", []))

    assert result.category == "식품"
    assert result.is_beverage is True


def test_classify_category_ignores_is_beverage_when_category_is_not_food(monkeypatch):
    """LLM이 규칙을 어기고 식품이 아닌 카테고리에 is_beverage=true를 보내도
    무시해야 한다(classify_category의 방어 로직)."""
    _install_fake_client(
        monkeypatch, '{"category": "가전디지털", "is_beverage": true}'
    )

    result = asyncio.run(category.classify_category("냉장고", []))

    assert result.category == "가전디지털"
    assert result.is_beverage is False


def test_classify_category_returns_none_for_unknown_category_value(monkeypatch):
    """목록에 없는 값을 LLM이 지어내면 category=None(분류 불확실)으로 안전하게
    처리해야 한다."""
    _install_fake_client(monkeypatch, '{"category": "없는카테고리", "is_beverage": false}')

    result = asyncio.run(category.classify_category("정체불명 상품", []))

    assert result.category is None


def test_classify_category_returns_none_on_api_failure(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("API 오류")

    class _BoomClient:
        chat = None

        def __init__(self):
            self.chat = self

        @property
        def completions(self):
            return self

        create = staticmethod(_boom)

    monkeypatch.setattr(category, "AsyncOpenAI", lambda **_: _BoomClient())

    result = asyncio.run(category.classify_category("아무거나", [SearchResult(title="t", url="https://a.com", snippet="s")]))

    assert result.category is None
    assert result.is_beverage is False
