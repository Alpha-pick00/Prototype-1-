"""Groq 전용 - 검색어 표기 변형 생성(2026-08-20, "2프로랑 2%랑 이프로랑 다
똑같은 제품인데 상품 매핑이 안되는 문제를 Groq을 통한 프롬프팅을 통해
해결해줘"). 11번가 검색 엔진이 사용자가 흔히 쓰는 표기("2프로")와 실제
카탈로그 표기("이프로"/"2%")가 달라 관련 상품을 하나도 못 찾는 경우가
있다(실측 확인 - "2프로"로 검색하면 카메라 삼각대·어댑터 같은 완전히
무관한 "프로"(Pro) 매칭 결과만 나온다) - 1차 검색이 관련 상품을 하나도
못 찾았을 때만(app.debate.run_elevenst_only_debate) 이 함수로 대안 표기를
만들어 재검색한다."""

from __future__ import annotations

from openai import AsyncOpenAI

from ..config import settings
from .base import parse_json_object

QUERY_VARIANT_INSTRUCTIONS = (
    "당신은 쇼핑 검색어의 다른 표기법을 제안하는 에이전트입니다. 주어진 검색어가 "
    "실제 상품 카탈로그에서는 다르게 표기될 수 있습니다(예: 숫자와 한글 혼용 "
    "표기 차이 - '2프로'/'이프로'/'2%', 띄어쓰기 차이, 흔한 오타/줄임말). "
    "이 검색어와 정확히 같은 상품을 가리키는 대안 표기를 최대 3개까지 "
    "제안하세요 - 다른 상품이나 브랜드를 제안하지 마세요, 표기만 다를 뿐 "
    "같은 것이어야 합니다. 확신이 없으면 빈 배열을 반환하세요. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{"variants": ["...", "..."]}'
)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_api_base, max_retries=0)


def build_query_variant_prompt(query: str) -> str:
    return f"{QUERY_VARIANT_INSTRUCTIONS}\n\n검색어: {query}"


async def generate_query_variants(query: str) -> list[str]:
    """실패(키 없음·API 오류·JSON 파싱 실패)하면 빈 리스트 - 호출부가 원래
    검색 결과 없음으로 그대로 처리한다."""
    if not settings.groq_api_key:
        return []
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": build_query_variant_prompt(query)}],
            response_format={"type": "json_object"},
        )
        data = parse_json_object(response.choices[0].message.content or "")
        variants = data.get("variants") or []
        return [str(v).strip() for v in variants if str(v).strip()][:3]
    except Exception:
        return []
