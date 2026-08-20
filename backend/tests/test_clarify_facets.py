"""AI 상세검색(2026-08-12) 테스트 - "음료수"처럼 짧고 애매한 검색어를 DeepSeek이
11번가 검색 결과 상품명에 근거해 facet(카테고리/브랜드/용량 등)으로 좁혀나가게
제안하는 기능(원래 Qwen으로 붙였다가 계정 활성화 문제로 DeepSeek로 옮겼다).
네트워크 요청 금지 - 전부 monkeypatch."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.debate import (
    _apply_category_breakdown,
    _category_breakdown_facet,
    _enrich_facets_per_brand,
    _MAX_BRAND_ENRICH_FANOUT,
    _select_category_group,
    _strip_query_answered_options,
    check_clarify_facets,
    run_elevenst_only_debate_stream,
)
from app.intent import is_non_product_chitchat, needs_clarification
from app.main import app
from app.schemas import ClarifyFacet

client = TestClient(app)


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
    async def _boom_search(query, limit=3):
        raise AssertionError("잡담 입력인데 elevenst.search_elevenst가 호출됐다")

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _boom_search)

    async def _boom_facets(query, names):
        raise AssertionError("잡담 입력인데 extract_facets_from_names가 호출됐다")

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _boom_facets)

    result = asyncio.run(check_clarify_facets("하이"))

    assert result.options.facets == []


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


def test_check_clarify_facets_attaches_facet_crossfilter_symmetrically(monkeypatch):
    """사용자 요청(2026-08-13: "삼성전자를 누르면은 시리즈에 삼성전자에 관한것만
    APPLE을 누르면 시리즈에 아이폰만" -> 2026-08-14: "시리즈에 초코파이 바나나를
    골랐다면 용량에 없는것들은 선택할수 없게" - 브랜드 전용이었던 걸 모든 facet
    쌍으로 일반화했다) - 검색을 다시 하지 않고, 이미 받아온 상품명만으로 옵션을
    다른 facet 값별로 미리 나눠서 응답에 실어줘야 한다. 양방향(브랜드->시리즈,
    시리즈->브랜드)으로 다 계산돼야 한다."""

    async def _fake_search_danawa(query, limit=3):
        return [
            {"pcode": "1", "product_name": "삼성전자 갤럭시S25 256GB", "total_mall_count": None},
            {"pcode": "2", "product_name": "삼성전자 갤럭시Z 폴드8 512GB", "total_mall_count": None},
            {"pcode": "3", "product_name": "APPLE 아이폰17 256GB", "total_mall_count": None},
            {"pcode": "4", "product_name": "APPLE 아이폰17 프로 512GB", "total_mall_count": None},
        ]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [
            ClarifyFacet(label="브랜드", options=["삼성전자", "APPLE"]),
            ClarifyFacet(label="시리즈", options=["갤럭시S25", "갤럭시Z 폴드8", "아이폰17", "아이폰17 프로"]),
        ]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("핸드폰"))

    by_label = {f.label: f for f in result.options.facets}
    assert by_label["브랜드"].options_by_selection == {
        "갤럭시S25": ["삼성전자"],
        "갤럭시Z 폴드8": ["삼성전자"],
        "아이폰17": ["APPLE"],
        "아이폰17 프로": ["APPLE"],
    }
    assert by_label["시리즈"].options_by_selection == {
        "삼성전자": ["갤럭시S25", "갤럭시Z 폴드8"],
        "APPLE": ["아이폰17", "아이폰17 프로"],
    }


def test_check_clarify_facets_crossfilter_works_between_non_brand_facets(monkeypatch):
    """사용자 요청(2026-08-14: "내가 만약 시리즈에 초코파이 바나나를 골랏다면
    용량에 없는것들은 선택할수없게 해야해") - 브랜드가 아니어도(시리즈 -> 용량)
    facet 사이 연관이 계산돼야 한다."""

    async def _fake_search_danawa(query, limit=3):
        return [
            {"pcode": "1", "product_name": "오리온 초코파이 바나나 468g", "total_mall_count": None},
            {"pcode": "2", "product_name": "오리온 초코파이 바나나 234g", "total_mall_count": None},
            {"pcode": "3", "product_name": "오리온 초코파이 오리지널 336g", "total_mall_count": None},
            {"pcode": "4", "product_name": "오리온 초코파이 오리지널 672g", "total_mall_count": None},
        ]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [
            ClarifyFacet(label="시리즈", options=["초코파이 바나나", "초코파이 오리지널"]),
            ClarifyFacet(label="용량", options=["468g", "234g", "336g", "672g"]),
        ]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("초코파이"))

    by_label = {f.label: f for f in result.options.facets}
    assert by_label["용량"].options_by_selection == {
        "초코파이 바나나": ["468g", "234g"],
        "초코파이 오리지널": ["336g", "672g"],
    }


def test_check_clarify_facets_orders_facets_from_macro_to_micro(monkeypatch):
    """사용자 요청(2026-08-14: "거시적인 선택에서 미시적인 선택으로 점차
    줄여나가게") - LLM이 낸 순서와 무관하게 카테고리/브랜드 같은 넓은 기준이
    용량/특징 같은 좁은 기준보다 먼저 오도록 정렬해야 한다."""

    async def _fake_search_danawa(query, limit=3):
        return [{"pcode": "1", "product_name": "오리온 초코파이 바나나 468g", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        # 일부러 미시적인 것부터 거꾸로 반환한다 - 정렬이 실제로 라벨 순서를
        # 바꾸는지 확인하려면 원래 순서가 이미 macro->micro면 안 된다.
        return [
            ClarifyFacet(label="특징", options=["저당", "고당"]),
            ClarifyFacet(label="용량", options=["468g", "234g"]),
            ClarifyFacet(label="브랜드", options=["오리온", "롯데"]),
        ]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("초코파이"))

    assert [f.label for f in result.options.facets] == ["브랜드", "용량", "특징"]


def test_check_clarify_facets_orders_phone_model_facet_first(monkeypatch):
    """사용자 요청(2026-08-14: "검색 순서에서 핸드폰 기종이 가장 먼저 위로
    올라가야할 것 같은데") - '핸드폰 기종' 기준은 카테고리/브랜드보다도 먼저
    와야 한다."""

    async def _fake_search_danawa(query, limit=3):
        return [{"pcode": "1", "product_name": "삼성전자 갤럭시S25 케이스", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names, required_labels=None):
        return [
            ClarifyFacet(label="브랜드", options=["삼성전자", "신지모루"]),
            ClarifyFacet(label="특징", options=["방수", "충격방지"]),
            ClarifyFacet(label="핸드폰 기종", options=["갤럭시S25", "갤럭시S26"]),
        ]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("핸드폰 케이스"))

    assert result.options.facets[0].label == "핸드폰 기종"


# -- 다나와 실측 카테고리 집계로 "카테고리" facet을 채우고, 선택된 카테고리로
# 표본을 다시 좁히는 기능(2026-08-18 사용자 리포트: "샤오미로 검색하면 AI
# 상세검색 카테고리에 휴대폰이 안 나온다" / "카테고리를 고르면 모델도, 모델을
# 고르면 용량도 그에 맞게 좁혀져야지 - 안 그러면 공기청정기 · 미 패드5처럼
# 존재하지 않는 상품으로 매핑돼") -------------------------------------------

_XIAOMI_CATEGORIES = [
    {
        "name": "태블릿/휴대폰",
        "count": 202586,
        "subcategories": [
            {"name": "휴대폰 주변용품", "count": 77372},
            {"name": "휴대폰", "count": 20469},
        ],
    },
    {
        "name": "생활가전",
        "count": 56381,
        "subcategories": [{"name": "청소기", "count": 47851}],
    },
]


def test_select_category_group_prefers_longest_match_on_overlap():
    # "태블릿/휴대폰"을 고른 뒤에도 "휴대폰"이라는 별개 중분류 이름이 그 문자열
    # 안에 부분적으로 걸린다 - 더 구체적인(긴) "태블릿/휴대폰"을 우선해야 한다.
    top = _select_category_group("샤오미 태블릿/휴대폰", _XIAOMI_CATEGORIES)
    assert top["name"] == "태블릿/휴대폰"


def test_category_breakdown_facet_uses_top_level_groups_by_default():
    facet = _category_breakdown_facet("샤오미", _XIAOMI_CATEGORIES)
    assert facet.label == "카테고리"
    assert facet.options == ["태블릿/휴대폰", "생활가전"]  # count 내림차순


def test_category_breakdown_facet_narrows_to_subcategories_once_selected():
    facet = _category_breakdown_facet("샤오미 태블릿/휴대폰", _XIAOMI_CATEGORIES)
    # 대분류 "태블릿/휴대폰"만 골랐을 뿐 중분류 "휴대폰"은 아직 안 골랐다 -
    # 대분류 이름 자체가 우연히 "휴대폰"을 부분 문자열로 포함한다고 해서
    # 이미 고른 것으로 착각해 옵션에서 지우면 안 된다(2026-08-18 실측:
    # 바로 이 문제로 정작 사용자가 원했던 "휴대폰" 옵션이 화면에서 사라졌었다).
    assert facet.options == ["휴대폰 주변용품", "휴대폰"]


def test_category_breakdown_facet_excludes_genuinely_chosen_subcategory():
    facet = _category_breakdown_facet("샤오미 태블릿/휴대폰 휴대폰", _XIAOMI_CATEGORIES)
    # 이번엔 "휴대폰"이 대분류 이름 바깥에 별도로 등장하므로 진짜로 고른 것 -
    # 남은 옵션이 "휴대폰 주변용품" 하나뿐이라(2개 미만) facet 자체가 사라진다.
    assert facet is None


def test_category_breakdown_facet_none_when_fewer_than_two_groups():
    assert _category_breakdown_facet("샤오미", [_XIAOMI_CATEGORIES[0]]) is None


def test_apply_category_breakdown_replaces_llm_facet_with_real_breakdown():
    # DeepSeek이 표본 편향(휴대폰이 표본에 없음)으로 "생활가전"만 뽑았어도,
    # 실측 집계에 있는 "태블릿/휴대폰"으로 교체돼야 한다.
    facets = [ClarifyFacet(label="카테고리", options=["생활가전"])]
    result = _apply_category_breakdown(facets, "샤오미", _XIAOMI_CATEGORIES)
    assert len(result) == 1
    assert result[0].options == ["태블릿/휴대폰", "생활가전"]


def test_apply_category_breakdown_inserts_when_no_category_facet_exists():
    facets = [ClarifyFacet(label="모델", options=["미지아 선풍기"])]
    result = _apply_category_breakdown(facets, "샤오미", _XIAOMI_CATEGORIES)
    assert [f.label for f in result] == ["카테고리", "모델"]


def test_check_clarify_facets_injects_real_category_facet_when_sample_misses_it(monkeypatch):
    """브랜드 전체 표본(다나와 검색 상위)에 특정 카테고리 상품이 하나도 없어도
    (샤오미 실측: 상위 40개가 전부 액세서리/가전이라 휴대폰이 없다), 다나와
    검색결과 페이지의 실측 카테고리 집계에 있으면 "카테고리" facet에 떠야
    한다."""

    async def _fake_search_danawa(query, limit=90):
        return [{"pcode": "1", "product_name": "샤오미 미지아 선풍기", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_search_danawa_categories(query):
        return _XIAOMI_CATEGORIES

    monkeypatch.setattr("fetchers.elevenst.search_categories", _fake_search_danawa_categories)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="모델", options=names)]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("샤오미"))

    by_label = {f.label: f for f in result.options.facets}
    assert "태블릿/휴대폰" in by_label["카테고리"].options


def test_check_clarify_facets_does_not_research_when_category_already_selected(monkeypatch):
    """HITL 구조적 필터 재설계(2026-08-20, "텍스트 재검색 말고 다른 방식으로") -
    카테고리를 이미 골랐어도(질의에 그 이름이 있어도) 그 이름을 검색어에
    덧붙여 11번가를 다시 검색하지 않는다(11번가는 카테고리 코드 필터를
    지원하지 않고, 카테고리 이름은 상품명 텍스트에 거의 등장하지 않아
    구조적 필터도 통하지 않는다) - 검색은 base_query로 한 번만 나가야
    한다."""
    seen_queries: list[str] = []

    async def _fake_search(query, limit=90):
        seen_queries.append(query)
        return [{"pcode": "1", "product_name": "샤오미 미지아 선풍기", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search)

    async def _fake_search_categories(query):
        return _XIAOMI_CATEGORIES

    monkeypatch.setattr("fetchers.elevenst.search_categories", _fake_search_categories)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="모델", options=names)]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    asyncio.run(check_clarify_facets("샤오미 태블릿/휴대폰 휴대폰", base_query="샤오미"))

    assert seen_queries == ["샤오미"]


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
    """구체적인 검색어는 needs_clarification()이 False라 다나와 검색조차 시도하지
    않아야 한다 - search_danawa가 불리면 바로 실패하도록 걸어서 확인한다."""

    async def _boom(query, limit=3):
        raise AssertionError("구체적인 검색어인데 elevenst.search_elevenst가 호출됐다")

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _boom)

    result = asyncio.run(check_clarify_facets("아이폰 15 프로 256기가"))

    assert result.options.facets == []


# -- check_clarify_facets: 정적 facet 캐시(속도 개선) --------------------------


def test_check_clarify_facets_uses_static_cache_without_any_search_or_llm_call(monkeypatch):
    """사용자 요청(2026-08-16: "'아이폰' 검색했을때... 그 AI상세검색하는 창...
    바로 띄워주라는 소리였어 - 질의검사하고 뭐하고 단계가 많으니까 그거를
    정규식으로 바꾸자") - facet_cache에 있는 카테고리는 검색도 DeepSeek 호출도
    없이 즉시 답해야 한다."""

    async def _boom_search(query, limit=3):
        raise AssertionError("정적 캐시에 있는 카테고리인데 elevenst.search_elevenst가 호출됐다")

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _boom_search)

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

    async def _fake_search_danawa(query, limit=3):
        seen.append(query)
        return [{"pcode": "1", "product_name": "아이폰 케이스 실리콘", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)
    monkeypatch.setattr(
        "app.agents.deepseek.extract_facets_from_names", lambda query, names: asyncio.sleep(0, result=[])
    )

    asyncio.run(check_clarify_facets("아이폰 케이스"))

    assert seen == ["아이폰 케이스"]


def test_check_clarify_facets_static_cache_miss_falls_through_to_real_search(monkeypatch):
    """목록에 없는 카테고리는 지금까지처럼 실제 검색+추출 경로를 그대로 타야 한다."""

    async def _fake_search_danawa(query, limit=3):
        return [{"pcode": "1", "product_name": "코카콜라 350ml 24개", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="카테고리", options=["탄산음료"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("과자"))

    assert result.options.facets == [ClarifyFacet(label="카테고리", options=["탄산음료"])]


def test_check_clarify_facets_returns_facets_for_ambiguous_query(monkeypatch):
    async def _fake_search_danawa(query, limit=3):
        return [
            {"pcode": "1", "product_name": "코카콜라 350ml 24개", "total_mall_count": None},
            {"pcode": "2", "product_name": "칠성사이다 190ml", "total_mall_count": None},
        ]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        assert names == ["코카콜라 350ml 24개", "칠성사이다 190ml"]
        return [ClarifyFacet(label="카테고리", options=["탄산음료"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("음료수"))

    assert result.mode == "clarify"
    assert result.options.facets == [ClarifyFacet(label="카테고리", options=["탄산음료"])]


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
    async def _fake_search_danawa(query, limit=3):
        return [
            {"pcode": "1", "product_name": "스탠리 퀜처 텀블러 887ml", "total_mall_count": None},
            {"pcode": "2", "product_name": "스탠리 아이스플로우 보틀 473ml", "total_mall_count": None},
        ]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="제품분류", options=["텀블러", "보틀"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("스탠리 텀블러"))

    assert result.options.facets == []


def test_check_clarify_facets_uses_wider_search_limit_than_the_fast_path(monkeypatch):
    """사용자 요청(2026-08-12: "브랜드가 2,3개 정도만 뜨는데") 회귀 테스트 -
    check_clarify_facets는 넓은 price_table.CLARIFY_SEARCH_LIMIT(90)을 써야
    한다(run_elevenst_only_debate의 좁은 limit=10과 다르게 - 상품명 표본이
    적으면 브랜드가 몇 개 안 뜬다)."""
    from app import price_table as price_table_module

    seen_limits: list[int] = []

    async def _fake_search_danawa(query, limit=3):
        seen_limits.append(limit)
        return []

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    asyncio.run(check_clarify_facets("음료수"))

    assert seen_limits == [price_table_module.CLARIFY_SEARCH_LIMIT]
    assert price_table_module.CLARIFY_SEARCH_LIMIT > 10


def test_check_clarify_facets_searches_base_query_instead_of_query(monkeypatch):
    """속도 개선(2026-08-13: "조금 더 빠르게") - base_query가 오면 그걸로
    검색해야 한다(캐시 재사용/Crawl-delay 회피가 목적) - query 그대로 검색하면
    드릴다운마다 매번 새 검색어라 캐시가 안 맞는다."""
    seen_queries: list[str] = []

    async def _fake_search_danawa(query, limit=3):
        seen_queries.append(query)
        return [
            {"pcode": "1", "product_name": "삼성전자 갤럭시S25 256GB", "total_mall_count": None},
            {"pcode": "2", "product_name": "삼성전자 갤럭시Z 폴드8 512GB", "total_mall_count": None},
            {"pcode": "3", "product_name": "삼성전자 갤럭시A57 128GB", "total_mall_count": None},
            {"pcode": "4", "product_name": "APPLE 아이폰17 256GB", "total_mall_count": None},
            {"pcode": "5", "product_name": "APPLE 아이폰17 프로 512GB", "total_mall_count": None},
        ]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="시리즈", options=names)]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("핸드폰 삼성전자", base_query="핸드폰"))

    assert seen_queries == ["핸드폰"]
    # base_query("핸드폰")의 넓은 표본에서 query("핸드폰 삼성전자")의 추가 단어
    # "삼성전자"로 로컬 필터링해야 하므로, APPLE 상품은 빠져야 한다(3개 남아
    # MIN_FILTERED_CLARIFY_ITEMS 이상이라 필터링이 그대로 적용된다).
    assert result.options.facets[0].options == [
        "삼성전자 갤럭시S25 256GB",
        "삼성전자 갤럭시Z 폴드8 512GB",
        "삼성전자 갤럭시A57 128GB",
    ]


def test_check_clarify_facets_enriches_minority_brand_series_via_per_brand_extraction(monkeypatch):
    """회귀 테스트(2026-08-13: "APLLE 을 선택했을때 시리즈 후보가 너무 적어") -
    한 번에 뽑으면 다수 브랜드(삼성전자)가 MAX_OPTIONS_PER_FACET 예산을 다 차지해
    소수 브랜드(APPLE) 시리즈가 아예 안 나올 수 있다. 브랜드별로 다시 뽑아서
    합쳐야 APPLE 시리즈도 온전히 나온다."""
    items = [
        {"pcode": "1", "product_name": "삼성전자 갤럭시S26 256GB", "total_mall_count": None},
        {"pcode": "2", "product_name": "삼성전자 갤럭시Z 폴드8 512GB", "total_mall_count": None},
        {"pcode": "3", "product_name": "APPLE 아이폰17 256GB", "total_mall_count": None},
    ]

    async def _fake_search_danawa(query, limit=3):
        return items

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

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
    하고 싶었어" -> "검색어 자체에 문제인거야..?") - 실측 결과 다나와 "핸드폰
    케이스" 검색 자체가 40개 중 갤럭시 36개/아이폰 1개로 쏠려 있었다. 표본
    안에서 아무리 잘 나눠도 원본에 아이폰 매물이 거의 없으면 소용없으므로,
    아이폰 표본이 부족하면(<3개) "아이폰 핸드폰 케이스"로 다나와에 보충 검색을
    한 번 더 돌려 진짜 아이폰 매물을 가져와야 한다."""
    base_items = [
        {"pcode": "1", "product_name": "갤럭시S26 케이스", "total_mall_count": None},
        {"pcode": "2", "product_name": "갤럭시Z 폴드8 케이스", "total_mall_count": None},
        {"pcode": "3", "product_name": "갤럭시S25 울트라 케이스", "total_mall_count": None},
        {"pcode": "4", "product_name": "아이폰17 케이스", "total_mall_count": None},
    ]
    # 보충 검색("아이폰 핸드폰 케이스")은 실제 다나와라면 아이폰 매물만 돌려준다
    # (갤럭시가 안 섞임) - 원래 검색(갤럭시 위주)과 구분해서 흉내낸다.
    iphone_supplement_items = [
        {"pcode": "5", "product_name": "아이폰17 케이스", "total_mall_count": None},
        {"pcode": "6", "product_name": "아이폰17 프로 케이스", "total_mall_count": None},
    ]

    async def _fake_search_danawa(query, limit=3):
        if "아이폰" in query:
            return iphone_supplement_items
        return base_items

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

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


def test_check_clarify_facets_falls_back_to_unfiltered_when_too_few_matches(monkeypatch):
    """필터링 결과가 너무 적으면(MIN_FILTERED_CLARIFY_ITEMS 미만) 필터링을
    포기하고 base_query의 넓은 표본을 그대로 쓴다 - 추가 검색은 하지 않는다."""

    async def _fake_search_danawa(query, limit=3):
        return [{"pcode": "1", "product_name": "삼성전자 갤럭시S25 256GB", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="시리즈", options=names)]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    result = asyncio.run(check_clarify_facets("핸드폰 없는브랜드", base_query="핸드폰"))

    # "없는브랜드"로 필터링하면 0개가 남아 MIN_FILTERED_CLARIFY_ITEMS(3) 미만이라
    # 필터링 전 표본(1개)을 그대로 써야 한다 - 빈 리스트가 되면 안 된다.
    assert result.options.facets[0].options == ["삼성전자 갤럭시S25 256GB"]


def test_check_clarify_facets_returns_empty_when_deepseek_finds_nothing(monkeypatch):
    async def _fake_search_danawa(query, limit=3):
        return [{"pcode": "1", "product_name": "테스트 상품", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)
    monkeypatch.setattr(
        "app.agents.deepseek.extract_facets_from_names", lambda query, names: asyncio.sleep(0, result=[])
    )

    result = asyncio.run(check_clarify_facets("테스트 상품"))

    assert result.options.facets == []


# -- POST /decide/clarify 엔드포인트 -------------------------------------------


def test_decide_clarify_endpoint_returns_clarify_response(monkeypatch):
    async def _fake_search_danawa(query, limit=3):
        return [{"pcode": "1", "product_name": "코카콜라 350ml", "total_mall_count": None}]

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _fake_search_danawa)

    async def _fake_extract_facets(query, names):
        return [ClarifyFacet(label="카테고리", options=["탄산음료", "주스"])]

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _fake_extract_facets)

    resp = client.post("/decide/clarify", json={"query": "음료수"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "clarify"
    assert data["options"]["facets"] == [
        {"label": "카테고리", "options": ["탄산음료", "주스"], "options_by_selection": None}
    ]


def test_decide_clarify_endpoint_empty_for_specific_query():
    resp = client.post("/decide/clarify", json={"query": "삼성전자 갤럭시 버즈3 프로"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["options"]["facets"] == []


# -- 회귀: run_elevenst_only_debate_stream은 짧은 검색어에도 LLM을 절대 안 부른다 ----


def test_run_elevenst_only_debate_stream_never_calls_deepseek_facets_even_for_short_query(monkeypatch):
    """check_clarify_facets()는 완전히 별도 진입점이고, run_elevenst_only_debate_stream()
    자체는 needs_clarification()을 아예 모른다 - "음료수" 같은 짧은 검색어를 이
    경로로 직접 태워도 extract_facets_from_names가 호출되면 안 된다(LLM 호출 0번
    불변식 유지 확인)."""

    async def _boom(query, names):
        raise AssertionError("run_elevenst_only_debate_stream이 facet 추출을 호출했다 - LLM 0회 불변식 위반")

    monkeypatch.setattr("app.agents.deepseek.extract_facets_from_names", _boom)

    async def _search_elevenst(query, limit=10):
        return []

    monkeypatch.setattr("fetchers.elevenst.search_elevenst", _search_elevenst)

    async def _collect():
        return [event async for event in run_elevenst_only_debate_stream("음료수")]

    events = asyncio.run(_collect())

    assert events == [
        {"type": "status", "stage": "searching"},
        {"type": "error", "message": "11번가에서 '음료수'에 대해 관련성 있는 상품을 찾지 못했다."},
    ]
