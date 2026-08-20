"""검색어의 카테고리를 자동으로 분류한다(2026-08-20, "카테고리는 선택안하고
쿼리를 기반으로 Groq이 자동으로 매핑할 수 있도록 해줘") - check_clarify_facets가
사용자에게 "카테고리"를 직접 고르게 하는 대신, 11번가 실측 카테고리 집계
(fetchers.elevenst.search_categories) 중 하나를 Groq이 질의만 보고 즉시
고르게 한다. 실패(키 없음·API 오류·범위 밖 index)하면 None - 호출부가
카테고리 없이도 계속 진행한다(카테고리는 어차피 표본을 좁히는 데 안 쓰인다,
app.debate.check_clarify_facets 참고)."""

from __future__ import annotations

from openai import AsyncOpenAI

from .agents.base import parse_json_object
from .config import settings

CLASSIFY_INSTRUCTIONS = (
    "당신은 쇼핑 검색어가 어떤 카테고리에 속하는지 분류하는 에이전트입니다. "
    "아래 후보 카테고리 목록 중에서 이 검색어에 가장 알맞은 것 하나만 "
    "고르세요 - 목록에 없는 카테고리를 새로 만들지 마세요. "
    "반드시 아래 후보 목록의 index 중 하나를 골라 JSON으로만 답하세요. "
    "다른 텍스트나 코드펜스를 덧붙이지 마세요.\n\n"
    '{"index": 0}'
)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_api_base, max_retries=0)


def build_classify_prompt(query: str, category_names: list[str]) -> str:
    options_block = "\n".join(f"[{i}] {name}" for i, name in enumerate(category_names))
    return f"{CLASSIFY_INSTRUCTIONS}\n\n검색어: {query}\n\n후보 카테고리:\n{options_block}"


async def classify_category(query: str, category_names: list[str]) -> str | None:
    if not category_names or not settings.groq_api_key:
        return None
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": build_classify_prompt(query, category_names)}],
            response_format={"type": "json_object"},
        )
        data = parse_json_object(response.choices[0].message.content or "")
        index = int(data.get("index"))
        if not (0 <= index < len(category_names)):
            return None
        return category_names[index]
    except Exception:
        return None
