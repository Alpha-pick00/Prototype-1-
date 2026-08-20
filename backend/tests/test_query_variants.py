"""run_elevenst_only_debate의 Groq 검색어 표기 변형 폴백
(_search_with_query_variants) 테스트 - "2프로"/"이프로"/"2%"처럼 사용자
표기와 11번가 카탈로그 표기가 달라 1차 검색이 관련 상품을 하나도 못 찾을
때만 쓰는 경로다. 네트워크 요청 금지 - 전부 monkeypatch."""

from __future__ import annotations

import asyncio

from app import debate
from app.agents import groq
from fetchers.elevenst import ElevenstSearchItem


def _item(name: str, price: int, code: str = "1") -> ElevenstSearchItem:
    return ElevenstSearchItem(
        product_code=code,
        product_name=name,
        price_krw=price,
        seller="판매자",
        url=f"https://www.11st.co.kr/products/{code}",
        review_count=None,
        buy_satisfy=None,
    )


def test_search_with_query_variants_returns_empty_when_groq_suggests_nothing(monkeypatch):
    async def _fake_variants(query):
        return []

    monkeypatch.setattr(debate.groq, "generate_query_variants", _fake_variants)

    async def _boom_search(query, limit=10):
        raise AssertionError("변형이 없는데 검색이 호출됐다")

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _boom_search)

    assert asyncio.run(debate._search_with_query_variants("2프로")) == []


def test_search_with_query_variants_matches_against_the_variant_not_original_query(monkeypatch):
    """실측: "2프로"↔"이프로 제품명" 유사도는 낮아(13점) 원래 질의로는 절대
    통과 못 한다 - 반드시 변형 문자열("이프로") 기준으로 관련성을 판정해야
    한다."""

    async def _fake_variants(query):
        assert query == "2프로"
        return ["2%", "이프로"]

    monkeypatch.setattr(debate.groq, "generate_query_variants", _fake_variants)

    async def _fake_search(query, limit=10):
        if query == "2%":
            return []
        if query == "이프로":
            return [_item("이프로 부족할때 제로 복숭아 500ml x 24개", 15000, "1")]
        raise AssertionError(f"예상 못한 검색어: {query}")

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search)

    relevant = asyncio.run(debate._search_with_query_variants("2프로"))

    assert [it["product_code"] for it in relevant] == ["1"]


def test_run_elevenst_only_debate_falls_back_to_query_variants_when_no_relevant_matches(monkeypatch):
    async def _fake_search(query, limit=10):
        if query == "2프로":
            return [_item("무관한 프로 카메라 삼각대", 30000, "irrelevant")]
        if query == "이프로":
            return [_item("이프로 부족할때 제로 복숭아 500ml x 24개", 15000, "1")]
        return []

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search)

    async def _fake_variants(query):
        return ["이프로"]

    monkeypatch.setattr(debate.groq, "generate_query_variants", _fake_variants)

    async def _no_embed(texts):
        return None

    monkeypatch.setattr(debate.embeddings, "embed", _no_embed)

    async def _no_recommend(query, candidates):
        return None

    monkeypatch.setattr(debate.gpt, "recommend_best", _no_recommend)

    result = asyncio.run(debate.run_elevenst_only_debate("2프로"))

    assert result.decision.product_name == "이프로 부족할때 제로 복숭아 500ml x 24개"


def test_run_elevenst_only_debate_still_fails_when_variants_also_find_nothing(monkeypatch):
    async def _fake_search(query, limit=10):
        return [_item("무관한 상품", 1000, "1")]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search)

    async def _fake_variants(query):
        return ["대안표기"]

    monkeypatch.setattr(debate.groq, "generate_query_variants", _fake_variants)

    try:
        asyncio.run(debate.run_elevenst_only_debate("무관한질의"))
        raise AssertionError("RuntimeError가 발생해야 한다")
    except RuntimeError as exc:
        assert "관련성 있는 상품을 찾지 못했다" in str(exc)


def test_generate_query_variants_returns_empty_when_key_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", None)

    assert asyncio.run(groq.generate_query_variants("2프로")) == []
