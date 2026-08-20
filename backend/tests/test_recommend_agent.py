"""run_elevenst_only_debate의 추천 Agent(임베딩 관련도 랭킹 + LLM 최종 선택)
테스트. 네트워크 요청 금지 - 전부 monkeypatch."""

from __future__ import annotations

import asyncio

from app import debate
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


def test_rank_by_relevance_orders_by_cosine_similarity_when_embeddings_available(monkeypatch):
    items = [_item("무관한 상품", 1000, "1"), _item("정확히 찾는 상품", 2000, "2")]

    async def _fake_embed(texts):
        # texts[0]는 query, 나머지는 후보 이름 순서대로 - "정확히 찾는 상품"만
        # query와 코사인 유사도 1.0이 되도록 벡터를 짠다.
        return [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]

    monkeypatch.setattr(debate.embeddings, "embed", _fake_embed)

    ranked = asyncio.run(debate._rank_by_relevance("정확히 찾는 상품", items))

    assert [it["product_code"] for it in ranked] == ["2", "1"]


def test_rank_by_relevance_falls_back_to_original_order_when_embedding_fails(monkeypatch):
    items = [_item("상품A", 1000, "1"), _item("상품B", 2000, "2")]

    async def _fake_embed(texts):
        return None

    monkeypatch.setattr(debate.embeddings, "embed", _fake_embed)

    ranked = asyncio.run(debate._rank_by_relevance("아무거나", items))

    assert [it["product_code"] for it in ranked] == ["1", "2"]


def test_run_elevenst_only_debate_uses_recommend_agent_pick_over_cheapest(monkeypatch):
    """추천 Agent가 최저가가 아닌 후보를 골라도(리뷰/구매만족도 등을 근거로)
    그 선택을 최종 추천으로 써야 한다 - 무조건 최저가를 강제하지 않는다."""

    async def _fake_search(query, limit=10):
        return [_item("찾는 상품 A", 1000, "1"), _item("찾는 상품 B", 2000, "2")]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search)
    monkeypatch.setattr(debate.price_table_module, "_product_name_matches", lambda a, b: True)

    async def _fake_embed(texts):
        return None  # 순서 그대로 유지

    monkeypatch.setattr(debate.embeddings, "embed", _fake_embed)

    async def _fake_recommend(query, candidates):
        return 1, "리뷰가 훨씬 많아서 더 신뢰할 만함"

    monkeypatch.setattr(debate.gpt, "recommend_best", _fake_recommend)

    result = asyncio.run(debate.run_elevenst_only_debate("찾는 상품"))

    assert result.decision.product_name == "찾는 상품 B"
    assert result.decision.price == "2,000원"
    assert "추천 Agent" in result.decision.reasoning
    assert len(result.proposals) == 2


def test_run_elevenst_only_debate_falls_back_to_cheapest_when_recommend_agent_fails(monkeypatch):
    async def _fake_search(query, limit=10):
        return [_item("찾는 상품 A", 2000, "1"), _item("찾는 상품 B", 1000, "2")]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search)
    monkeypatch.setattr(debate.price_table_module, "_product_name_matches", lambda a, b: True)

    async def _fake_embed(texts):
        return None

    monkeypatch.setattr(debate.embeddings, "embed", _fake_embed)

    async def _fake_recommend(query, candidates):
        return None

    monkeypatch.setattr(debate.gpt, "recommend_best", _fake_recommend)

    result = asyncio.run(debate.run_elevenst_only_debate("찾는 상품"))

    assert result.decision.product_name == "찾는 상품 B"
    assert result.decision.price == "1,000원"
    assert "최저가" in result.decision.reasoning


def test_recommend_best_returns_none_for_out_of_range_index(monkeypatch):
    from app.agents import gpt

    class _FakeMessage:
        content = '{"index": 5, "reasoning": "아무거나"}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(gpt, "_client", lambda: _FakeClient())

    result = asyncio.run(gpt.recommend_best("질의", [{"product_name": "상품", "price_krw": 1000, "seller": "s"}]))

    assert result is None


def test_recommend_best_returns_index_and_reasoning_on_success(monkeypatch):
    from app.agents import gpt

    class _FakeMessage:
        content = '{"index": 0, "reasoning": "리뷰가 많음"}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(gpt, "_client", lambda: _FakeClient())

    result = asyncio.run(gpt.recommend_best("질의", [{"product_name": "상품", "price_krw": 1000, "seller": "s"}]))

    assert result == (0, "리뷰가 많음")
