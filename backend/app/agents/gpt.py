from openai import AsyncOpenAI

from ..config import settings
from ..schemas import (
    BrandOption,
    BulkProposal,
    JudgeVerdict,
    SearchResult,
)
from .base import (
    build_brand_price_prompt,
    build_bulk_prompt,
    build_clarify_ask_prompt,
    build_relaxed_pick_prompt,
    filter_bulk_options,
    is_danawa_comparison_page,
    is_generic_listing_url,
    parse_json_array,
    parse_json_object,
)

# 이 모듈이 담당하는 에이전트 슬롯은 스키마/프론트엔드/테스트 전반에서
# agent="gpt"로 식별된다(파일명·함수명도 그대로) - 하지만 실제로 호출하는
# 모델은 2026-08-15부터 OpenAI가 아니라 Qwen이다(사용자 요청: "GPT 토큰이
# 더 이상 없어서 Qwen 성능 제일 좋은 걸로 바꿔줘"). DashScope가 OpenAI SDK와
# 호환되는 엔드포인트를 제공해서, openai SDK를 base_url만 바꿔 그대로 쓴다
# (agents/deepseek.py와 동일한 패턴). agent="gpt" 식별자 자체를 "qwen"으로
# 바꾸지 않은 이유 - AgentName 리터럴, DB에 저장된 과거 기록, 프론트엔드 타입,
# 테스트 픽스처 등 수십 곳에 걸쳐 있어 그 리네임 자체가 훨씬 큰(그리고 지금
# 급한 문제와 무관한) 변경이 된다. 사용자에게 보이는 이름만
# frontend/src/app/components/SearchResults.tsx의 AGENT_LABEL에서 "Qwen"으로
# 바꿔뒀다.


def _client() -> AsyncOpenAI:
    # max_retries=0 - 사용자 요청(2026-08-15: "너무
    # 느려 더 빠르게"). 실패해도 호출부가 이미 폴백을 갖고 있어 SDK 재시도로
    # 얻는 이득보다 지연 비용이 크다.
    return AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_api_base, max_retries=0)


async def propose_bulk(query: str, search_results: list[SearchResult]) -> BulkProposal:
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.qwen_model,
            messages=[{"role": "user", "content": build_bulk_prompt(query, search_results)}],
        )
        options = parse_json_array(response.choices[0].message.content or "")
        options = filter_bulk_options(options, search_results)
        return BulkProposal(agent="gpt", options=options)
    except Exception as exc:
        return BulkProposal(agent="gpt", error=str(exc))


_CLARIFY_ASK_FALLBACK = "몇 가지 후보를 찾았어요 — 아래에서 골라주시겠어요?"


async def generate_clarify_question(query: str, options: list[str]) -> str:
    """이번 라운드에 물어봐야 할 축(브랜드/제품/용량/개수)의 후보들을 실제
    상담원처럼 자연스러운 한 질문으로 바꾼다 — 프론트가 "브랜드를 선택하면
    좁혀드려요" 같은 고정 라벨 대신 이 문장을 채팅 말풍선으로 보여준다.
    호출 실패 시 고정 안내 문구로 대체한다."""
    if not options:
        return _CLARIFY_ASK_FALLBACK
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.qwen_model,
            messages=[{"role": "user", "content": build_clarify_ask_prompt(query, options)}],
            response_format={"type": "json_object"},
        )
        data = parse_json_object(response.choices[0].message.content or "")
        return data.get("message") or _CLARIFY_ASK_FALLBACK
    except Exception:
        return _CLARIFY_ASK_FALLBACK


async def find_lowest_price(
    query: str, brand: str, search_results: list[SearchResult]
) -> BrandOption | None:
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.qwen_model,
            messages=[
                {"role": "user", "content": build_brand_price_prompt(query, brand, search_results)}
            ],
            response_format={"type": "json_object"},
        )
        data = parse_json_object(response.choices[0].message.content or "")
        if not data.get("product_name"):
            return None
        if is_generic_listing_url(data.get("url", "")):
            return None
        return BrandOption(brand=brand, **data)
    except Exception:
        return None


async def pick_most_relevant(query: str, search_results: list[SearchResult]) -> JudgeVerdict | None:
    """정확히 일치하는 후보가 하나도 없을 때의 폴백(2026-08-15, "적절한 상품
    후보를 찾지 못하면 다시 fallback해서 feedback 구조로 돌아가서 가장
    관련성 높은 상품을 추천해주는 시스템") - adk_pipeline.run_stream()이
    일반 propose/challenge/judge 경로에서 후보를 하나도 못 건졌을 때만
    호출한다. 완벽히 일치하지 않아도 검색 결과 중 가장 관련성 높은 것을
    고르되, reasoning에 왜 완벽히 일치하지 않는지를 먼저 밝히게 해서 UI가
    "낮은 확신" 캐비어로 보여줄 수 있게 한다(build_relaxed_pick_prompt 참고).
    그마저도 없으면(검색 결과 전체가 아예 무관) None."""
    if not search_results:
        return None
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.qwen_model,
            messages=[{"role": "user", "content": build_relaxed_pick_prompt(query, search_results)}],
            response_format={"type": "json_object"},
        )
        data = parse_json_object(response.choices[0].message.content or "")
        if not data.get("product_name") or not data.get("url"):
            return None
        url = data.get("url", "")
        if is_generic_listing_url(url) or is_danawa_comparison_page(url):
            return None
        return JudgeVerdict(**data)
    except Exception:
        return None
