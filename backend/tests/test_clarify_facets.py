"""AI 상세검색(2026-08-12) 테스트 - "음료수"처럼 짧고 애매한 검색어를 DeepSeek이
검색 결과 상품명에 근거해 facet(브랜드/용량 등)으로 좁혀나가게 제안하는 기능
(원래 Qwen으로 붙였다가 계정 활성화 문제로 DeepSeek로 옮겼다).

(2026-08-20) check_clarify_facets()의 검색 백엔드를 다나와 직접 스크래핑에서
11번가(app.search.search)로 옮겼다 - 메인 파이프라인(adk_pipeline)이 먼저
11번가로 전환됐는데 이 함수만 남아있던 걸 뒤늦게 맞췄다. base_query 재사용/
카테고리 표본 좁히기 최적화는 다나와의 느린 검색(Crawl-delay)을 우회하려던
용도라 11번가에선 필요 없어져 제거했다 - 관련 테스트도 함께 삭제했다.
네트워크 요청 금지 - 전부 monkeypatch."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app import decision_cache
from app.debate import (
    _enrich_facets_per_brand,
    _MAX_BRAND_ENRICH_FANOUT,
    _strip_query_answered_options,
    check_clarify_facets,
    run_clarify,
    run_danawa_only_debate_stream,
    run_debate,
    run_debate_stream,
)
from app.intent import is_non_product_chitchat, needs_clarification
from app.main import app
from app.schemas import ClarifyFacet, Decision, DecideResponse, SearchResult

client = TestClient(app)


def _sr(title: str, url: str | None = None) -> SearchResult:
    """11번가 검색 결과 하나를 흉내낸다 - url을 안 주면 title만으로 안전한
    (제네릭 목록 페이지로 안 걸리는) 상품 상세 URL을 만든다."""
    return SearchResult(title=title, url=url or f"https://www.11st.co.kr/products/{abs(hash(title))}", snippet="", score=None)


# -- intent.needs_clarification: 짧고 숫자 없는 검색어 휴리스틱 -----------------


def test_needs_clarification_true_for_short_bare_category_word():
    assert needs_clarification("음료수") is True


def test_needs_clarification_true_for_two_word_bare_query():
    assert needs_clarification("과자 선물") is True


def test_needs_clarification_false_for_query_with_digit():
    # "테스트 상품 15" 처럼 숫자가 섞이면 이미 구체적인 스펙 검색으로 본다.
    assert needs_clarification("아이폰 15") is False


def test_needs_clarification_false_for_long_specific_query():
    assert needs_clarification("삼성전자 갤럭시 버즈3 프로 그래파이트") is False


def test_needs_clarification_false_for_bulk_spec_query():
    # 단위/수량이 붙으면 is_bulk_query가 우선이라 clarify로 새지 않는다(기존 동작).
    assert needs_clarification("생수 500ml") is False


def test_needs_clarification_still_true_for_buy_intent_phrase():
    # 기존(2026-08-10 이전) 동작 - "사고싶다"류 문구는 길이/숫자와 무관하게 그대로 유지.
    assert needs_clarification("이거 진짜 사고 싶은데 뭐가 좋을까") is True


# -- intent.is_non_product_chitchat: 인사말/잡담 즉시 감지(속도 개선) -------------


def test_is_non_product_chitchat_true_for_bare_greeting():
    assert is_non_product_chitchat("하이") is True
    assert is_non_product_chitchat("안녕하세요") is True
    assert is_non_product_chitchat("Hi") is True
    assert is_non_product_chitchat("ㅋㅋㅋ") is True


def test_is_non_product_chitchat_false_for_real_short_product_query():
    # "테스트 상품"은 기존 테스트 스위트에서 "못 찾은 상품 검색어"로 쓰이는
    # 문구다 - 잡담으로 오탐하면 안 된다(needs_clarification은 여전히 True).
    assert is_non_product_chitchat("테스트 상품") is False
    assert is_non_product_chitchat("음료수") is False
    assert is_non_product_chitchat("아이폰 15") is False


def test_is_non_product_chitchat_false_when_greeting_word_is_substring():
    # 전체 문자열이 인사말과 정확히 일치할 때만 True - 부분 문자열은 오탐하지 않는다.
    assert is_non_product_chitchat("하이마트 에어컨") is False


def test_is_non_product_chitchat_true_for_pronoun_or_question_opener():
    # 닫힌 인사말 집합 밖의, 봇에게 말을 거는 임의의 잡담/시비도 잡아야 한다
    # (사용자 요청: "'하이' '안녕' 이것만 처리해놨네 ... 다른 쓸데없는 말 하니까
    # 왜이리 오래걸려").
    assert is_non_product_chitchat("너 뒤질래") is True
    assert is_non_product_chitchat("너 뭐야") is True
    assert is_non_product_chitchat("왜 이렇게 비싸") is True
    assert is_non_product_chitchat("누구세요") is True
    assert is_non_product_chitchat("심심하다") is True


def test_is_non_product_chitchat_false_for_pronoun_prefix_that_is_a_real_product():
    # 접두사 매칭이었다면 "너"로 시작한다는 이유로 오탐됐을 실제 상품명들 -
    # 첫 토큰이 "너"/"장어" 등과 정확히 일치할 때만 판정하므로 안전해야 한다.
    assert is_non_product_chitchat("너구리") is False
    assert is_non_product_chitchat("너구리 라면") is False
    assert is_non_product_chitchat("휴지") is False
    assert is_non_product_chitchat("장어") is False


def test_is_non_product_chitchat_false_for_long_sentence():
    # 잡담 판정은 짧은 문장에만 적용된다 - 길면 진짜 구매 의도/상세 설명일
    # 가능성이 높아 보수적으로 접는다.
    assert is_non_product_chitchat("너 혹시 이 근처에서 제일 싸게 파는 데 아는 곳 있어?") is False


def test_is_non_product_chitchat_false_for_buy_intent_even_with_chitchat_shape():
    # BUY_INTENT_PATTERN이 먼저 적용돼야 한다 - 구매 의도 문구는 잡담이 아니다.
    assert is_non_product_chitchat("이거 진짜 사고 싶은데 뭐가 좋을까") is False


# -- 회귀: 잡담 입력은 검색/LLM 호출 없이 즉시 실패한다(속도 개선) -----------------


def test_check_clarify_facets_returns_empty_immediately_for_greeting(monkeypatch):
    async def _boom_search(query, max_results=20):
        raise AssertionError("잡담 입력인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom_search)

    async def _boom_facets(query, names):
        raise AssertionError("잡담 입력인데 extract_facets_from_names가 호출됐다")

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _boom_facets)

    result = asyncio.run(check_clarify_facets("하이"))

    assert result.options.facets == []


def test_run_debate_stream_fails_fast_for_greeting_without_any_search_or_llm_call(monkeypatch):
    async def _boom_search(query, max_results=12):
        raise AssertionError("잡담 입력인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom_search)
    monkeypatch.setattr("app.debate._any_llm_key_configured", lambda: True)

    async def _collect():
        return [event async for event in run_debate_stream("안녕하세요")]

    events = asyncio.run(_collect())

    assert events == [{"type": "error", "message": "적절한 상품 후보를 찾지 못했습니다."}]


def test_run_debate_raises_immediately_for_greeting(monkeypatch):
    async def _boom_search(query, max_results=12):
        raise AssertionError("잡담 입력인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom_search)
    monkeypatch.setattr("app.debate._any_llm_key_configured", lambda: True)

    try:
        asyncio.run(run_debate("하이"))
        raise AssertionError("RuntimeError가 발생해야 한다")
    except RuntimeError as exc:
        assert str(exc) == "적절한 상품 후보를 찾지 못했습니다."


# -- 정적 최종결과 캐시(속도 개선) -------------------------------------------


def test_decision_cache_lookup_matches_regardless_of_token_order():
    """AI 상세검색에서 facet을 클릭하는 순서가 달라도(dedupeAppend가 다른
    순서로 이어붙여도) 같은 선택 집합이면 같은 캐시 항목을 찾아야 한다."""
    forward = decision_cache.lookup("아이폰 17 256GB 자급제")
    shuffled = decision_cache.lookup("자급제 256GB 아이폰 17")

    assert forward is not None
    assert forward == shuffled


def test_decision_cache_lookup_returns_none_for_unknown_combo():
    assert decision_cache.lookup("아이폰 아이폰 아이폰") is None


def test_run_debate_stream_uses_static_decision_cache_without_full_pipeline(monkeypatch):
    """사용자 요청(2026-08-16: "여기서 상세검색까지 누르면 바로 정규식으로
    찾을수있게 0.1초만에 해줘") - 캐시에 있는 facet 조합이면 정제/검색/제안/
    검증/심사 전체를 건너뛰고 즉시 최종 결과를 내야 한다."""

    async def _boom(query, max_results=12):
        raise AssertionError("정적 캐시에 있는 조합인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom)
    monkeypatch.setattr("app.debate._any_llm_key_configured", lambda: True)

    async def _collect():
        return [event async for event in run_debate_stream("아이폰 17 256GB 자급제")]

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert events[0]["type"] == "final"
    assert events[0]["result"]["decision"]["product_name"]


def test_run_debate_uses_static_decision_cache_without_full_pipeline(monkeypatch):
    async def _boom(query, max_results=12):
        raise AssertionError("정적 캐시에 있는 조합인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom)
    monkeypatch.setattr("app.debate._any_llm_key_configured", lambda: True)

    result = asyncio.run(run_debate("아이폰 17 256GB 자급제"))

    assert isinstance(result, DecideResponse)
    assert result.decision.product_name


# -- 회귀: 잡담 판정을 못 빠져나간 인식 불가 입력도 검색이 완전히 비면 즉시 실패한다 ----


def test_run_clarify_fails_fast_when_search_finds_nothing_without_full_pipeline(monkeypatch):
    """is_non_product_chitchat이 못 잡는 임의의 인식 불가 텍스트라도, 11번가
    검색 자체가 아무것도 못 찾았으면 run_single_debate(정제+검색+제안+검증+
    심사 전체 재실행)까지 새지 않고 바로 실패해야 한다."""

    async def _empty_search(query, max_results=10):
        return []

    monkeypatch.setattr("app.search.search", _empty_search)

    async def _no_options(query, results, persona=None):
        return None

    monkeypatch.setattr("app.debate._extract_clarify_options", _no_options)

    async def _boom_single_debate(query, skip_clarify=False, persona=None):
        raise AssertionError("검색 결과가 0개인데 run_single_debate까지 흘러갔다")

    monkeypatch.setattr("app.debate.run_single_debate", _boom_single_debate)

    try:
        asyncio.run(run_clarify("완전히 인식 불가능한 문자열입니다아아"))
        raise AssertionError("RuntimeError가 발생해야 한다")
    except RuntimeError as exc:
        assert str(exc) == "적절한 상품 후보를 찾지 못했습니다."


def test_run_clarify_still_falls_back_to_full_pipeline_when_search_has_results(monkeypatch):
    """검색 결과가 있는데 clarify 옵션만 못 뽑았으면(기존 동작) 여전히
    run_single_debate로 폴백해야 한다 - 이번 변경으로 이 경로를 막으면 안 된다."""

    async def _some_results(query, max_results=10):
        return [_sr("11번가 상품")]

    monkeypatch.setattr("app.search.search", _some_results)

    async def _no_options(query, results, persona=None):
        return None

    monkeypatch.setattr("app.debate._extract_clarify_options", _no_options)

    called = {"value": False}

    async def _fake_single_debate(query, skip_clarify=False, persona=None):
        called["value"] = True
        return DecideResponse(
            query=query,
            proposals=[],
            decision=Decision(
                product_name="상품", price="1,000원", retailer="쿠팡",
                url="https://coupang.com/vp/products/1", reasoning="근거", chosen_agent="gpt",
            ),
        )

    monkeypatch.setattr("app.debate.run_single_debate", _fake_single_debate)

    asyncio.run(run_clarify("음료수"))

    assert called["value"] is True


# -- app.agents.deepseek.extract_facets_from_names ---------------------------


def test_extract_facets_from_names_parses_deepseek_json_response(monkeypatch):
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"카테고리": ["탄산음료", "주스", "생수"], "용량": ["500ml", "1.5L"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    facets = asyncio.run(deepseek.extract_facets_from_names("음료수", ["코카콜라 350ml", "칠성사이다 190ml"]))

    assert len(facets) == 2
    labels = {f.label for f in facets}
    assert labels == {"카테고리", "용량"}


def test_extract_facets_from_names_sorts_brand_options_by_popularity(monkeypatch):
    """사용자 요청(2026-08-12: "브랜드도 인기순으로 정렬") - LLM이 알려준 순서를
    그대로 믿지 않고, 실제 상품명에 몇 번 등장하는지로 다시 정렬해야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        # LLM은 "매일유업"을 먼저 말했지만, 실제 상품명에는 "롯데칠성음료"가 더 많이 등장한다.
        content = '{"facets": {"브랜드": ["매일유업", "롯데칠성음료"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = [
        "롯데칠성음료 칠성사이다 190ml",
        "롯데칠성음료 펩시 500ml",
        "롯데칠성음료 밀키스 250ml",
        "매일유업 초코우유 200ml",
    ]
    facets = asyncio.run(deepseek.extract_facets_from_names("음료수", names))

    assert len(facets) == 1
    assert facets[0].options == ["롯데칠성음료", "매일유업"]


def test_extract_facets_from_names_allows_more_brand_options_than_other_facets(monkeypatch):
    """사용자 요청(2026-08-12: "브랜드가 2,3개 정도만 뜨는데 ... 찾기 기능도
    있었으면") - 브랜드/제조사 기준은 다른 기준(상한 6개)보다 훨씬 넓게(15개까지) 보여준다."""
    from app.agents import deepseek

    many_brands = [f"브랜드{i}" for i in range(20)]
    many_volumes = [f"{i}00ml" for i in range(20)]

    class _FakeMessage:
        content = f'{{"facets": {{"브랜드": {many_brands!r}, "용량": {many_volumes!r}}}}}'.replace("'", '"')

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    facets = asyncio.run(deepseek.extract_facets_from_names("음료수", ["상품 1"]))

    by_label = {f.label: f for f in facets}
    assert len(by_label["브랜드"].options) == 15
    assert len(by_label["용량"].options) == 6


def test_extract_facets_from_names_drops_facets_with_only_one_distinct_option(monkeypatch):
    """사용자 요청(2026-08-13: "카테고리에 스마트폰은 있으면 안되고") - 값이
    하나뿐인 기준은 골라도 아무것도 안 좁혀지니 애초에 응답에서 빠져야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"카테고리": ["스마트폰"], "브랜드": ["삼성전자", "APPLE"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰", ["삼성전자 갤럭시S25", "APPLE 아이폰17"]))

    labels = {f.label for f in facets}
    assert labels == {"브랜드"}


def test_extract_facets_from_names_strips_purchase_type_terms_from_container_form(monkeypatch):
    """사용자 리포트(2026-08-14: 음료 검색에서 용기형태 선택지로 "업소용"이
    나옴 - 페트/캔이 나와야 정상) - LLM이 구매유형 수식어를 용기형태로 잘못
    묶어 보내도, "업소용" 같은 알려진 비-용기형태 값은 코드에서 걸러내야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"용기형태": ["업소용", "페트", "캔"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = ["코카콜라 업소용 페트 1.5L", "코카콜라 캔 250ml"]
    facets = asyncio.run(deepseek.extract_facets_from_names("콜라", names))

    assert len(facets) == 1
    assert facets[0].label == "용기형태"
    assert "업소용" not in facets[0].options
    assert set(facets[0].options) == {"페트", "캔"}


def test_extract_facets_from_names_drops_container_form_facet_when_only_purchase_type_terms(monkeypatch):
    """용기형태로 뽑힌 값 전부가 알려진 비-용기형태 값이면(필터 후 1개 이하만
    남으면), 애초에 값이 하나뿐인 기준과 동일하게 그 facet 자체를 버려야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"용기형태": ["업소용", "가정용"], "브랜드": ["코카콜라", "펩시"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    facets = asyncio.run(deepseek.extract_facets_from_names("콜라", ["코카콜라 업소용", "펩시 가정용"]))

    labels = {f.label for f in facets}
    assert labels == {"브랜드"}


def test_extract_facets_from_names_strips_non_purchase_type_values(monkeypatch):
    """사용자 리포트(2026-08-14: "핸드폰 케이스" 검색에서 구매유형으로 "해외"/
    "중고"가 뜸 - 상품명에 그런 단어가 없는데도 DeepSeek이 스마트폰 시장 통념을
    끌어와 만들어냄) - "구매유형" 라벨의 값 중 알려진 구매유형 어휘가 아닌 값은
    코드에서 걸러내야 한다(용기형태와 반대로 화이트리스트 방식)."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"구매유형": ["정품", "리퍼", "해외", "아이폰15"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = ["아이폰15 케이스 정품", "아이폰15 케이스 리퍼"]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    assert len(facets) == 1
    assert facets[0].label == "구매유형"
    assert "해외" not in facets[0].options
    assert "아이폰15" not in facets[0].options
    assert set(facets[0].options) == {"정품", "리퍼"}


def test_extract_facets_from_names_drops_purchase_type_facet_when_no_known_terms(monkeypatch):
    """구매유형으로 뽑힌 값 전부가 알려진 구매유형 어휘가 아니면(필터 후 0개면),
    값이 하나뿐인 기준과 동일하게 그 facet 자체를 버려야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"구매유형": ["아이폰6", "아이폰15"], "브랜드": ["APPLE", "삼성전자"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    facets = asyncio.run(
        deepseek.extract_facets_from_names("핸드폰 케이스", ["APPLE 아이폰6 케이스", "삼성전자 케이스"])
    )

    labels = {f.label for f in facets}
    assert labels == {"브랜드"}


def test_extract_facets_from_names_drops_value_that_is_substring_of_another_in_same_facet(monkeypatch):
    """실측 사례(2026-08-14: "핸드폰 케이스" 검색에서 "부가기능" 기준에 "생활방수"와
    별개로 "방수"만 단독으로도 뜸) - 한 값이 같은 기준의 다른 값에 이미 완전히
    포함되는 부분 문자열이면 독자적인 선택지가 아니므로 버려야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"부가기능": ["생활방수", "방수", "카드수납"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = ["삼성전자 케이스 생활방수", "삼성전자 케이스 카드수납"]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    assert len(facets) == 1
    assert "방수" not in facets[0].options
    assert set(facets[0].options) == {"생활방수", "카드수납"}


def test_extract_facets_from_names_filters_out_phone_models_older_than_2020(monkeypatch):
    """사용자 요청(2026-08-14: "2020년 이후 모델로만 보이게 하는 방법 없어?
    아이폰 12부터라던지") - 아이폰/갤럭시S/갤럭시노트는 세대 번호가 출시
    연도와 거의 그대로 대응하므로(아이폰12=2020, 갤럭시S20=2020,
    갤럭시노트20=2020) 그보다 이전 세대는 '핸드폰 기종'에서 빼야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = (
            '{"facets": {"핸드폰 기종": '
            '["아이폰17", "아이폰11", "갤럭시S25", "갤럭시S10", "갤럭시노트20", "갤럭시노트9"]}}'
        )

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = [
        "아이폰17 케이스", "아이폰11 케이스",
        "갤럭시S25 케이스", "갤럭시S10 케이스",
        "갤럭시노트20 케이스", "갤럭시노트9 케이스",
    ]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    by_label = {f.label: f for f in facets}
    options = set(by_label["핸드폰 기종"].options)
    assert options == {"아이폰17", "갤럭시S25", "갤럭시노트20"}
    assert "아이폰11" not in options
    assert "갤럭시S10" not in options
    assert "갤럭시노트9" not in options


def test_extract_facets_from_names_does_not_recency_filter_families_without_a_reliable_rule(monkeypatch):
    """갤럭시Z(폴드/플립)·갤럭시A·아이패드는 세대 번호가 연식과 느슨하게만
    대응해 안전한 컷오프 규칙이 없다 - 걸러야 할 값을 놓치더라도 최신 값을
    잘못 지우지 않도록, 이 계열은 연식 필터를 적용하지 않는다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = '{"facets": {"핸드폰 기종": ["갤럭시A10", "갤럭시Z 폴드2", "아이패드 프로"]}}'

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = ["갤럭시A10 케이스", "갤럭시Z 폴드2 케이스", "아이패드 프로 케이스"]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    by_label = {f.label: f for f in facets}
    assert set(by_label["핸드폰 기종"].options) == {"갤럭시A10", "갤럭시Z 폴드2", "아이패드 프로"}


def test_extract_facets_from_names_keeps_value_only_in_first_facet_that_claims_it(monkeypatch):
    """실측 사례(2026-08-14: "핸드폰 케이스" 검색에서 "맥세이프"가 "기종"에도
    "특징"에도 동시에 뜸) - 같은 값이 여러 기준에 동시에 뜨면 먼저 나온 기준이
    차지하고 이후 기준에서는 빠져야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = (
            '{"facets": {'
            '"기종": ["맥세이프", "마그네틱", "갤럭시S25", "갤럭시S26"], '
            '"특징": ["맥세이프", "방수", "충격방지"]'
            "}}"
        )

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = [
        "맥세이프 마그네틱 갤럭시S25 케이스",
        "마그네틱 갤럭시S26 케이스",
        "맥세이프 방수 충격방지 케이스",
    ]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    by_label = {f.label: f for f in facets}
    assert set(by_label["핸드폰 기종"].options) == {"갤럭시S25", "갤럭시S26"}
    assert "맥세이프" in by_label["기종"].options
    assert "맥세이프" not in by_label["특징"].options
    assert set(by_label["특징"].options) == {"방수", "충격방지"}


def test_extract_facets_from_names_consolidates_all_device_brands_into_single_phone_model_facet(monkeypatch):
    """사용자 요청(2026-08-14: "갤럭시 전용 이렇게 없애고, 핸드폰 기종별로
    선택할 수 있게") - 갤럭시/아이폰 모델명이 어느 라벨에 담겨 왔든 기기
    브랜드로 나누지 않고 '핸드폰 기종' 기준 하나로 합쳐야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = (
            '{"facets": {'
            '"카테고리": ["케이스", "스탠드", "갤럭시S25 울트라", "아이폰17", "아이폰17 프로"], '
            '"특징": ["마그넷", "방수", "갤럭시Z 폴드8"]'
            "}}"
        )

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = [
        "갤럭시S25 울트라 케이스 마그넷",
        "갤럭시Z 폴드8 스탠드 방수",
        "아이폰17 케이스",
        "아이폰17 프로 케이스",
    ]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    by_label = {f.label: f for f in facets}
    assert set(by_label.keys()) == {"핸드폰 기종", "카테고리", "특징"}
    assert set(by_label["핸드폰 기종"].options) == {
        "갤럭시S25 울트라",
        "아이폰17",
        "아이폰17 프로",
        "갤럭시Z 폴드8",
    }
    assert set(by_label["카테고리"].options) == {"케이스", "스탠드"}
    assert set(by_label["특징"].options) == {"마그넷", "방수"}


def test_extract_facets_from_names_keeps_brand_facet_alongside_phone_model_facet(monkeypatch):
    """사용자 요청(2026-08-14: "제조사는 그대로 넣어도 될 것 같아 다시 살려줘" -
    바로 앞서 "제조사는 필요없을 것 같고"라며 뺐던 걸 되돌림) - '핸드폰 기종'
    기준이 있어도 '브랜드'/'제조사' 기준을 지우지 않고 그대로 둬야 한다."""
    from app.agents import deepseek

    class _FakeMessage:
        content = (
            '{"facets": {'
            '"브랜드": ["삼성전자", "신지모루", "슈피겐"], '
            '"핸드폰 기종": ["갤럭시S25", "갤럭시S26"]'
            "}}"
        )

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    names = ["삼성전자 갤럭시S25 케이스", "신지모루 갤럭시S26 케이스", "슈피겐 갤럭시S25 케이스"]
    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    by_label = {f.label: f for f in facets}
    assert set(by_label.keys()) == {"브랜드", "핸드폰 기종"}
    assert set(by_label["브랜드"].options) == {"삼성전자", "신지모루", "슈피겐"}
    assert set(by_label["핸드폰 기종"].options) == {"갤럭시S25", "갤럭시S26"}


def test_extract_facets_from_names_balances_phone_model_options_across_brand_ecosystems(monkeypatch):
    """사용자 리포트(2026-08-14: "선택지에는 너무 갤럭시만 모여서 보여주는
    경향이있어" - 아이폰14를 고르려 해도 목록에 없어서 직접 입력해야 함) -
    표본에 갤럭시 매물이 압도적으로 많으면(20종, 각 3회 등장) 아이폰(2종, 각
    1회 등장)은 순수 인기순 정렬로는 상한(15개) 안에 전혀 못 들어간다 - 브랜드
    facet 쏠림을 브랜드별 재추출로 푼 것과 같은 원리로, '핸드폰 기종'은 계열별
    라운드로빈으로 뽑아 아이폰도 최소한 일부는 포함되게 해야 한다."""
    from app.agents import deepseek

    galaxy_models = [f"갤럭시S25 {i}" for i in range(20)]
    iphone_models = ["아이폰17", "아이폰17 프로"]

    class _FakeMessage:
        content = f'{{"facets": {{"핸드폰 기종": {galaxy_models + iphone_models!r}}}}}'.replace("'", '"')

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

    monkeypatch.setattr(deepseek, "_client", lambda: _FakeClient())

    # 갤럭시 모델은 각 3회, 아이폰 모델은 각 1회만 등장 - 순수 인기순 정렬이면
    # 갤럭시(count=3)가 전부 아이폰(count=1)보다 위로 가 상한 15개를 다 차지한다.
    names = [f"{m} 케이스" for m in galaxy_models for _ in range(3)] + [f"{m} 케이스" for m in iphone_models]

    facets = asyncio.run(deepseek.extract_facets_from_names("핸드폰 케이스", names))

    by_label = {f.label: f for f in facets}
    options = by_label["핸드폰 기종"].options
    assert len(options) == 15  # MAX_BRAND_OPTIONS 상한 그대로 채워짐
    assert "아이폰17" in options
    assert "아이폰17 프로" in options


def test_extract_facets_from_names_returns_empty_on_no_product_names():
    from app.agents import deepseek

    facets = asyncio.run(deepseek.extract_facets_from_names("음료수", []))
    assert facets == []


def test_extract_facets_from_names_swallows_client_errors(monkeypatch):
    from app.agents import deepseek

    class _BoomClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("API 키 없음")

    monkeypatch.setattr(deepseek, "_client", lambda: _BoomClient())

    facets = asyncio.run(deepseek.extract_facets_from_names("음료수", ["코카콜라 350ml"]))
    assert facets == []


# -- app.debate.check_clarify_facets ------------------------------------------


def test_check_clarify_facets_skips_search_for_specific_query(monkeypatch):
    """구체적인 검색어는 needs_clarification()이 False라 11번가 검색조차 시도하지
    않아야 한다 - search가 불리면 바로 실패하도록 걸어서 확인한다."""

    async def _boom(query, max_results=20):
        raise AssertionError("구체적인 검색어인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom)

    result = asyncio.run(check_clarify_facets("아이폰 15 프로 256기가"))

    assert result.options.facets == []


# -- check_clarify_facets: 정적 facet 캐시(속도 개선) --------------------------


def test_check_clarify_facets_uses_static_cache_without_any_search_or_llm_call(monkeypatch):
    """사용자 요청(2026-08-16: "'아이폰' 검색했을때... 그 AI상세검색하는 창...
    바로 띄워주라는 소리였어 - 질의검사하고 뭐하고 단계가 많으니까 그거를
    정규식으로 바꾸자") - facet_cache에 있는 카테고리는 검색도 DeepSeek 호출도
    없이 즉시 답해야 한다."""

    async def _boom_search(query, max_results=20):
        raise AssertionError("정적 캐시에 있는 카테고리인데 search가 호출됐다")

    monkeypatch.setattr("app.search.search", _boom_search)

    async def _boom_facets(query, names):
        raise AssertionError("정적 캐시에 있는 카테고리인데 extract_facets_from_names가 호출됐다")

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _boom_facets)

    result = asyncio.run(check_clarify_facets("아이폰"))

    assert result.mode == "clarify"
    assert len(result.options.facets) > 0


def test_check_clarify_facets_static_cache_ignores_queries_with_extra_words(monkeypatch):
    """"아이폰 케이스"처럼 카테고리 키워드를 포함하지만 실제로는 다른 걸 찾는
    질의까지 아이폰 facet으로 잘못 가로채면 안 된다 - 전체 질의가 정확히
    일치할 때만 정적 캐시를 쓴다(부분 문자열 매치 아님)."""
    seen: list[str] = []

    async def _fake_search(query, max_results=20):
        seen.append(query)
        return [_sr("아이폰 케이스 실리콘")]

    monkeypatch.setattr("app.search.search", _fake_search)
    monkeypatch.setattr(
        "app.agents.deepseek.extract_facets_from_names", lambda query, names: asyncio.sleep(0, result=[])
    )

    asyncio.run(check_clarify_facets("아이폰 케이스"))

    assert seen == ["아이폰 케이스"]


def test_check_clarify_facets_static_cache_miss_falls_through_to_real_search(monkeypatch):
    """목록에 없는 카테고리는 지금까지처럼 실제 검색+추출 경로를 그대로 타야 한다."""

    async def _fake_search(query, max_results=20):
        return [_sr("코카콜라 350ml 24개")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="브랜드", options=["오리온"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("과자"))

    assert result.options.facets == [ClarifyFacet(label="브랜드", options=["오리온"])]


# -- 대화체 질의 정제 (2026-08-20, "'안녕 충전기 살래' 했는데도 적절한 상품을 -----
# 못찾았다" 리포트 - adk_pipeline의 refine과 별개로 이 함수도 자체적으로
# 정제해야 했다) -----------------------------------------------------------


def test_check_clarify_facets_refines_conversational_query_before_searching(monkeypatch):
    """"안녕 충전기 살래"처럼 인사말/구매의도 문구가 섞인 질의는 groq.refine_query로
    정제한 뒤에야 검색에 써야 한다 - 원본 그대로 검색하면 잡음 때문에 실제
    상품을 잘 못 찾는다."""
    from app.agents import groq

    captured_search_query: list[str] = []

    async def _fake_search(query, max_results=20):
        captured_search_query.append(query)
        return [_sr("삼성전자 25W 고속충전기")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_refine_query(query):
        assert query == "안녕 충전기 살래"
        return "충전기"

    monkeypatch.setattr(groq, "refine_query", _fake_refine_query)

    async def _fake_extract_facets(query, names):
        return []

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    asyncio.run(check_clarify_facets("안녕 충전기 살래"))

    assert captured_search_query == ["충전기"]


def test_check_clarify_facets_skips_refine_for_already_clean_query(monkeypatch):
    """"과자"처럼 이미 짧고 깨끗한 검색어는 groq.refine_query를 아예 호출하지
    않아야 한다 - 매번 불렀다면 이번 세션에서 줄인 LLM 호출 수가 다시 늘어난다."""
    from app.agents import groq

    async def _fake_search(query, max_results=20):
        return [_sr("오리온 초코파이")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fail_if_called(query):
        raise AssertionError("이미 깨끗한 검색어인데 refine_query가 호출됐다")

    monkeypatch.setattr(groq, "refine_query", _fail_if_called)

    async def _fake_extract_facets(query, names):
        return []

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    asyncio.run(check_clarify_facets("과자"))


def test_check_clarify_facets_returns_facets_for_ambiguous_query(monkeypatch):
    async def _fake_search(query, max_results=20):
        return [_sr("코카콜라 350ml 24개"), _sr("칠성사이다 190ml")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_extract_facets(query, names):
        assert names == ["코카콜라 350ml 24개", "칠성사이다 190ml"]
        return [ClarifyFacet(label="브랜드", options=["코카콜라", "칠성사이다"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("음료수"))

    assert result.mode == "clarify"
    assert result.options.facets == [ClarifyFacet(label="브랜드", options=["코카콜라", "칠성사이다"])]


def test_check_clarify_facets_strips_category_facet_even_if_deepseek_still_proposes_one(monkeypatch):
    """(2026-08-20 실측 회귀) 프롬프트에서 "카테고리"를 예시/JSON 형식에서 빼고
    만들지 말라고 명시했는데도(agents.base.FACET_CLARIFY_INSTRUCTIONS) DeepSeek이
    "초코파이" 검색에 "카테고리": ["초코파이", "과자세트", "과자", ...]처럼
    검색어 자체를 되묻는 facet을 스스로 만들어낸 사례가 있었다(다른 프롬프트
    지시를 안정적으로 안 지키는 _strip_query_answered_options와 같은 유형의 문제).
    _extract_facets가 label=="카테고리"인 facet을 한 번 더 걸러내는지 확인한다."""

    async def _fake_search(query, max_results=20):
        return [_sr("오리온 초코파이 바나나 468g")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_extract_facets(query, names):
        return [
            ClarifyFacet(label="카테고리", options=["초코파이", "과자세트", "과자", "선물세트", "파이", "케이크"]),
            ClarifyFacet(label="용량", options=["468g", "234g"]),
        ]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("초코파이"))

    by_label = {f.label: f for f in result.options.facets}
    assert "카테고리" not in by_label
    assert by_label["용량"].options == ["468g", "234g"]


def test_strip_query_answered_options_removes_value_already_in_query():
    """사용자 리포트(2026-08-18 "스탠리 텀블러 검색했는데 물어보는 게 반복되고
    많다") 회귀 테스트 - 검색어에 이미 있는 단어("텀블러")를 facet이 선택지로
    또 보여주면 이미 답한 걸 다시 묻는 것처럼 느껴진다."""
    facets = [
        ClarifyFacet(
            label="제품분류",
            options=["텀블러", "보틀", "머그"],
            options_by_selection={"473ml": ["텀블러", "보틀"], "709ml": ["텀블러"]},
        )
    ]

    result = _strip_query_answered_options("스탠리 텀블러", facets)

    assert result == [
        ClarifyFacet(
            label="제품분류",
            options=["보틀", "머그"],
            options_by_selection={"473ml": ["보틀"]},
        )
    ]


def test_strip_query_answered_options_drops_facet_left_with_under_two_values():
    """필터링 후 서로 다른 값이 1개 이하로 남으면 그 기준 자체가 더 이상 좁혀주는
    게 없으므로 facet 전체를 뺀다."""
    facets = [ClarifyFacet(label="제품분류", options=["텀블러", "보틀"])]

    result = _strip_query_answered_options("스탠리 텀블러 보틀", facets)

    assert result == []


def test_strip_query_answered_options_leaves_untouched_facet_with_only_one_option():
    """필터링으로 걸러진 게 하나도 없으면, 그 facet이 원래부터 옵션 1개뿐이었어도
    이 함수가 임의로 지우면 안 된다(그건 추출 쪽 책임)."""
    facets = [ClarifyFacet(label="시리즈", options=["삼성전자 갤럭시S25 256GB"])]

    result = _strip_query_answered_options("핸드폰 없는브랜드", facets)

    assert result == facets


def test_check_clarify_facets_strips_query_redundant_option_end_to_end(monkeypatch):
    async def _fake_search(query, max_results=20):
        return [_sr("스탠리 퀜처 텀블러 887ml"), _sr("스탠리 아이스플로우 보틀 473ml")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="제품분류", options=["텀블러", "보틀"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("스탠리 텀블러"))

    assert result.options.facets == []


def test_check_clarify_facets_uses_its_own_search_limit(monkeypatch):
    """check_clarify_facets는 11번가 search_cache.FETCH_SIZE(20)만큼 검색해
    facet 추출 표본을 최대한 넓게 잡아야 한다(이보다 크게 요청해도 캐시
    크기 이상은 못 받는다)."""
    from app import debate

    seen_limits: list[int] = []

    async def _fake_search(query, max_results=20):
        seen_limits.append(max_results)
        return []

    monkeypatch.setattr("app.search.search", _fake_search)

    asyncio.run(check_clarify_facets("음료수"))

    assert seen_limits == [debate._CLARIFY_SEARCH_LIMIT]


def test_check_clarify_facets_enriches_minority_brand_series_via_per_brand_extraction(monkeypatch):
    """회귀 테스트(2026-08-13: "APLLE 을 선택했을때 시리즈 후보가 너무 적어") -
    한 번에 뽑으면 다수 브랜드(삼성전자)가 MAX_OPTIONS_PER_FACET 예산을 다 차지해
    소수 브랜드(APPLE) 시리즈가 아예 안 나올 수 있다. 브랜드별로 다시 뽑아서
    합쳐야 APPLE 시리즈도 온전히 나온다."""
    items = [
        _sr("삼성전자 갤럭시S26 256GB"),
        _sr("삼성전자 갤럭시Z 폴드8 512GB"),
        _sr("APPLE 아이폰17 256GB"),
    ]

    async def _fake_search(query, max_results=20):
        return items

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_extract_facets(query, names, required_labels=None):
        # 이 가짜 LLM은 "삼성전자 상품명만 들어오면" 삼성 시리즈만 뽑고(원래
        # 문제 상황 재현), 브랜드별로 좁혀 다시 부른 호출(required_labels가 옴)은
        # 그 안에 있는 브랜드만 반영한다 - 실제 DeepSeek이 브랜드가 섞인 채로
        # 부르면 다수 브랜드가 예산을 다 차지하는 상황을 흉내낸다.
        has_apple = any("apple" in n.lower() for n in names)
        has_samsung = any("삼성전자" in n for n in names)
        if required_labels:
            # 브랜드별 재추출 - required_labels(그대로 재사용해야 하는 라벨)를 지킨다.
            if has_apple and not has_samsung:
                return [ClarifyFacet(label=required_labels[0], options=["아이폰17"])]
            if has_samsung:
                return [ClarifyFacet(label=required_labels[0], options=["갤럭시S26", "갤럭시Z 폴드8"])]
            return []
        facets = [ClarifyFacet(label="브랜드", options=["삼성전자", "APPLE"])]
        if has_samsung:
            facets.append(ClarifyFacet(label="시리즈", options=["갤럭시S26", "갤럭시Z 폴드8"]))
        return facets

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("핸드폰"))

    by_label = {f.label: f for f in result.options.facets}
    # 원래 결합 호출(전체 상품명, 삼성 우세)로는 "아이폰17"이 안 나왔어야 하지만,
    # APPLE 전용 재추출 덕분에 병합돼 있어야 한다.
    assert "아이폰17" in by_label["시리즈"].options
    assert by_label["시리즈"].options_by_selection is not None
    assert by_label["시리즈"].options_by_selection["APPLE"] == ["아이폰17"]


def test_enrich_facets_per_brand_caps_parallel_llm_calls(monkeypatch):
    """토큰 절약(2026-08-19) - 브랜드가 MAX_BRAND_OPTIONS(15)까지 있어도
    _enrich_facets_per_brand는 상위 _MAX_BRAND_ENRICH_FANOUT개까지만 DeepSeek를
    병렬 호출해야 한다(요청 한 번에 최대 15번 부르던 걸 상한을 둬 줄인 회귀
    테스트)."""
    many_brands = [f"브랜드{i}" for i in range(10)]
    assert len(many_brands) > _MAX_BRAND_ENRICH_FANOUT

    facets = [
        ClarifyFacet(label="브랜드", options=many_brands),
        ClarifyFacet(label="시리즈", options=["시리즈A"]),
    ]
    names = [f"{b} 상품" for b in many_brands]

    calls: list[str] = []

    async def _fake_extract_facets(query, names, required_labels=None):
        calls.append(names[0] if names else "")
        return []

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    asyncio.run(_enrich_facets_per_brand(facets, names, "질의"))

    assert len(calls) == _MAX_BRAND_ENRICH_FANOUT


def test_check_clarify_facets_enriches_minority_ecosystem_device_models_via_ecosystem_extraction(monkeypatch):
    """사용자 리포트(2026-08-14: "갤럭시랑 아이폰이랑 비슷한 비율로 기종이 뜨게
    하고 싶었어" -> "검색어 자체에 문제인거야..?") - 실측 결과 검색 자체가 40개
    중 갤럭시 36개/아이폰 1개로 쏠려 있었다. 표본 안에서 아무리 잘 나눠도
    원본에 아이폰 매물이 거의 없으면 소용없으므로, 아이폰 표본이 부족하면
    (<3개) "아이폰 핸드폰 케이스"로 다나와에 보충 검색을 한 번 더 돌려 진짜
    아이폰 매물을 가져와야 한다(_extract_facets 안의 _enrich_device_models_by_ecosystem
    은 이 세션에서도 다나와 보충 검색 그대로 쓴다 - check_clarify_facets 자체의
    주 검색만 11번가로 옮겼다)."""
    base_items = [_sr("갤럭시S26 케이스"), _sr("갤럭시Z 폴드8 케이스"), _sr("갤럭시S25 울트라 케이스"), _sr("아이폰17 케이스")]
    # 보충 검색("아이폰 핸드폰 케이스")은 다나와 직접 검색 경로(_ecosystem_name_pool)를
    # 그대로 타므로, 그 경로만 danawa_search를 모킹한다.
    iphone_supplement_items = [
        {"pcode": "5", "product_name": "아이폰17 케이스", "total_mall_count": None},
        {"pcode": "6", "product_name": "아이폰17 프로 케이스", "total_mall_count": None},
    ]

    async def _fake_search(query, max_results=20):
        return base_items

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_search_danawa(query, limit=3):
        assert "아이폰" in query
        return iphone_supplement_items

    monkeypatch.setattr("fetchers.danawa_search.search_danawa", _fake_search_danawa)

    async def _fake_extract_facets(query, names, required_labels=None):
        has_iphone = any("아이폰" in n for n in names)
        has_galaxy = any("갤럭시" in n for n in names)
        if required_labels:
            # 기종 생태계별 재추출 - required_labels(그대로 재사용해야 하는 라벨)를 지킨다.
            if has_iphone and not has_galaxy:
                models = ["아이폰17"]
                if any("프로" in n for n in names):
                    models.append("아이폰17 프로")
                return [ClarifyFacet(label=required_labels[0], options=models)]
            if has_galaxy:
                return [
                    ClarifyFacet(
                        label=required_labels[0],
                        options=["갤럭시S26", "갤럭시Z 폴드8", "갤럭시S25 울트라"],
                    )
                ]
            return []
        # 결합 호출은 갤럭시 매물이 많아 갤럭시만 뽑는다(원래 버그 재현) - 아이폰17은 못 뽑음.
        return [ClarifyFacet(label="핸드폰 기종", options=["갤럭시S26", "갤럭시Z 폴드8", "갤럭시S25 울트라"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("핸드폰 케이스"))

    by_label = {f.label: f for f in result.options.facets}
    options = by_label["핸드폰 기종"].options
    # 원래 결합 호출(전체 상품명, 갤럭시 우세)로는 "아이폰17"이 안 나왔어야 하지만,
    # 보충 검색으로 찾은 "아이폰17 프로"까지 병합돼 있어야 한다(원래 표본엔
    # 아이폰17만 있었으므로, "아이폰17 프로"가 있다는 건 보충 검색이 실제로
    # 새 데이터를 가져왔다는 증거다).
    assert "아이폰17" in options
    assert "아이폰17 프로" in options


def test_check_clarify_facets_returns_empty_when_deepseek_finds_nothing(monkeypatch):
    async def _fake_search(query, max_results=20):
        return [_sr("테스트 상품")]

    monkeypatch.setattr("app.search.search", _fake_search)
    monkeypatch.setattr(
        "app.agents.deepseek.extract_facets_from_names", lambda query, names: asyncio.sleep(0, result=[])
    )

    result = asyncio.run(check_clarify_facets("테스트 상품"))

    assert result.options.facets == []


# -- POST /decide/clarify 엔드포인트 -------------------------------------------


def test_decide_clarify_endpoint_returns_clarify_response(monkeypatch):
    async def _fake_search(query, max_results=20):
        return [_sr("코카콜라 350ml")]

    monkeypatch.setattr("app.search.search", _fake_search)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="브랜드", options=["코카콜라", "칠성사이다"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    resp = client.post("/decide/clarify", json={"query": "음료수"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarify"
    assert data["options"]["facets"] == [
        {"label": "브랜드", "options": ["코카콜라", "칠성사이다"], "options_by_selection": None}
    ]


def test_decide_clarify_endpoint_empty_for_specific_query():
    resp = client.post("/decide/clarify", json={"query": "삼성전자 갤럭시 버즈3 프로"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["options"]["facets"] == []


# -- 회귀: run_danawa_only_debate*는 짧은 검색어에도 여전히 LLM을 절대 안 부른다 ----


def test_run_danawa_only_debate_stream_never_calls_deepseek_facets_even_for_short_query(monkeypatch):
    """check_clarify_facets()는 완전히 별도 진입점이고, run_danawa_only_debate_stream()
    자체는 needs_clarification()을 아예 모른다 - "음료수" 같은 짧은 검색어를 이
    경로로 직접 태워도 extract_facets_from_names가 호출되면 안 된다(LLM 호출 0번
    불변식 유지 확인 - 이 경로 자체는 deepseek.propose 등 다른 LLM 호출도 원래
    안 하지만, 이 테스트는 새로 추가한 facet 추출 쪽만 특정해서 확인한다). 이
    경로는 다나와 실측 가격표만 쓰는 별도 실험 경로라 여전히 다나와 직접
    검색을 쓴다(check_clarify_facets의 11번가 전환과 무관)."""

    async def _boom(query, names):
        raise AssertionError("run_danawa_only_debate_stream이 facet 추출을 호출했다 - LLM 0회 불변식 위반")

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _boom)

    async def _search_danawa(query, limit=3):
        return []

    monkeypatch.setattr("fetchers.danawa_search.search_danawa", _search_danawa)

    async def _collect():
        return [event async for event in run_danawa_only_debate_stream("음료수")]

    events = asyncio.run(_collect())

    assert events == [
        {"type": "error", "message": "다나와에서 '음료수'에 대한 가격 정보를 찾지 못했다(검색/실측 모두 실패)."}
    ]


# -- 회귀: run_debate()의 LLM 키 미설정 우선순위 -------------------------------


def test_run_debate_routes_to_danawa_only_when_no_llm_key_even_for_short_query(monkeypatch):
    """2026-08-12에 needs_clarification()을 넓히면서 드러난 순서 버그의 회귀
    테스트 - LLM 키가 하나도 없으면(_any_llm_key_configured False) "테스트 상품"
    처럼 이제 clarify로도 보이는 짧은 검색어라도 run_clarify(facet 추출 호출)로
    새지 않고 그대로 run_danawa_only_debate로 가야 한다. 이 실험 경로는
    다나와 실측 가격표만 쓴다(check_clarify_facets의 11번가 전환과 무관)."""
    monkeypatch.setattr("app.debate._any_llm_key_configured", lambda: False)

    async def _boom_facets(query, product_names, required_labels=None):
        raise AssertionError("LLM 키가 없는데 deepseek.extract_facets_from_names이 호출됐다")

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _boom_facets)

    async def _search_danawa(query, limit=3):
        return []

    monkeypatch.setattr("fetchers.danawa_search.search_danawa", _search_danawa)

    try:
        asyncio.run(run_debate("테스트 상품"))
    except RuntimeError as exc:
        # 다나와 실측 데이터가 없어 못 찾았다는 정상적인 실패 - run_danawa_only_debate까지
        # 도달했다는 뜻이므로 이 테스트의 목적(run_clarify로 안 샜는지)엔 이걸로 충분하다.
        assert "가격 정보를 찾지 못했다" in str(exc)
