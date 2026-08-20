"""app/intent.py의 순수 함수(로컬 휴리스틱, 네트워크/LLM 호출 없음) 테스트."""

from __future__ import annotations

from app.intent import looks_conversational_query


# -- looks_conversational_query (2026-08-20, "'안녕 나 컵을 사고싶어' 이런식으로 -----
# 쿼리를 입력하면 지금 LLM이 못 알아듣거든" 회귀 - refine을 조건부로 재도입할 때
# 이 함수가 그 트리거 기준이 된다) ------------------------------------------------


def test_looks_conversational_query_true_for_greeting_plus_buy_intent():
    assert looks_conversational_query("안녕 나 컵을 사고싶어") is True


def test_looks_conversational_query_true_for_greeting_prefix_with_request():
    assert looks_conversational_query("안녕하세요 카메라 좀 추천해주세요") is True


def test_looks_conversational_query_true_for_buy_intent_without_greeting():
    assert looks_conversational_query("이거 사고 싶어") is True


def test_looks_conversational_query_false_for_short_bare_query():
    """"음료수"처럼 이미 짧고 깨끗한 검색어는 정제가 필요 없다 - refine을
    매번 태우면 이번 세션에서 줄인 LLM 호출 수가 다시 늘어난다."""
    assert looks_conversational_query("음료수") is False


def test_looks_conversational_query_false_for_specific_query():
    assert looks_conversational_query("삼성전자 갤럭시 버즈3 프로") is False


def test_looks_conversational_query_false_for_pure_greeting_alone():
    """인사말 하나뿐인 순수 잡담(예: "안녕하세요")은 is_non_product_chitchat이
    이미 앞단에서 걸러내므로 여기서 True일 필요가 없다."""
    assert looks_conversational_query("안녕하세요") is False


def test_looks_conversational_query_false_for_empty_string():
    assert looks_conversational_query("") is False
    assert looks_conversational_query("   ") is False
