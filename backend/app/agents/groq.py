import logging

from openai import AsyncOpenAI

from ..config import settings
from ..schemas import BulkProposal, SearchResult
from .base import build_bulk_prompt, build_refine_query_prompt, filter_bulk_options, parse_json_array, parse_json_object

logger = logging.getLogger(__name__)

# 이 모듈이 담당하는 에이전트 슬롯은 스키마/프론트엔드/테스트 전반에서
# agent="groq"로 식별된다(파일명·함수명도 그대로). 원래는 실제로 Google Gemini를
# 호출했었지만(그래서 파일명이 agents/gemini.py였다), 2026-08-16부터 Gemini
# 프로젝트가 403으로 막혀 Groq로 전환했다(사용자 요청: "deepseek Qwen 빼고 싹 다
# 무료 모델로 바꾸려고 해"). 처음엔 "gpt" 슬롯(agents/gpt.py, Qwen으로 전환된
# 뒤에도 식별자를 그대로 둔 것)과 같은 이유로 agent="gemini" 식별자를 유지했는데,
# 2026-08-18("Gemini 이제 안쓰니까 이름 제대로 바꿔서 코드 반영해") 사용자가
# 실제로 안 쓰는 이름을 그대로 두는 것보다 지금 쓰는 모델명으로 바로잡는 걸
# 선택해 agent="groq"로 리네임했다 - 스키마(AgentName)·프론트엔드·테스트
# 전반의 참조도 함께 바꿨다.
#
# 2026-08-21("기업에서 hcx api 를 제공을 해줘가지고 ... hcx로 바꿔줘") 실제
# 호출 대상을 다시 Naver Cloud CLOVA Studio(HCX)로 바꿨다 - 이번엔 이름을
# 바꿔달라는 요청이 없어 "groq" 식별자/모듈명은 그대로 둔다(gemini→groq
# 리네임과 달리). HCX도 OpenAI 호환 엔드포인트를 제공해서(api_key 하나만
# 있으면 되는 Bearer 인증, 별도 헤더 불필요) openai SDK를 base_url만 바꿔
# 그대로 쓴다(agents/deepseek.py와 동일한 패턴) - 자세한 내용은
# config.py의 groq_api_key/groq_api_base 주석 참고.


def _client() -> AsyncOpenAI:
    # max_retries=0 - 사용자 요청(2026-08-15: "너무
    # 느려 더 빠르게"). 실패해도 호출부가 이미 폴백을 갖고 있어 SDK 재시도로
    # 얻는 이득보다 지연 비용이 크다.
    return AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_api_base, max_retries=0)


async def propose_bulk(query: str, search_results: list[SearchResult]) -> BulkProposal:
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": build_bulk_prompt(query, search_results)}],
        )
        options = parse_json_array(response.choices[0].message.content or "")
        options = filter_bulk_options(options, search_results)
        return BulkProposal(agent="groq", options=options)
    except Exception as exc:
        return BulkProposal(agent="groq", error=str(exc))


async def refine_query(query: str) -> str:
    """대화체/인사말이 섞인 질의를 실제 검색어로 정제한다(2026-08-20, "안녕
    충전기 살래"가 적절한 상품을 못 찾는 리포트) - app.debate.check_clarify_facets
    (AI 상세검색)는 adk_pipeline의 refine LlmAgent를 거치지 않는 완전히 별도
    경로라, 그쪽에 조건부로 재도입한 refine과 별개로 이 함수가 필요했다.
    같은 프롬프트(REFINE_QUERY_INSTRUCTIONS)와 같은 모델(groq_refine_model -
    구조화 출력 지원, adk_pipeline._build_refine_agent 참고)을 그대로 재사용해
    두 경로의 정제 결과가 어긋나지 않게 한다. 실패하면(API 오류, JSON 파싱
    실패, 빈 응답 등) 원본 질의를 그대로 돌려준다 - 정제 실패가 검색 자체를
    막으면 안 된다(호출부가 이미 "정제 없이도 원본으로 계속 진행" 가능하도록
    짜여 있다).

    response_format={"type": "json_object"}는 Groq 시절에 쓰던 강제 JSON
    모드였는데, 2026-08-21 HCX로 전환하면서 뺐다 - HCX의 OpenAI 호환
    엔드포인트는 response_format.type=json_schema만 문서화돼 있고 json_object는
    확인되지 않아(api.ncloud-docs.com), 이 모듈 밖의 다른 agents 모듈
    (judge.py/category.py/ocr/cleanup.py)과 똑같이 프롬프트 지시
    (REFINE_QUERY_INSTRUCTIONS의 "반드시 JSON 형식으로만 답하라") +
    parse_json_object 파싱만으로 맞춘다."""
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.groq_refine_model,
            messages=[{"role": "user", "content": build_refine_query_prompt(query)}],
        )
        data = parse_json_object(response.choices[0].message.content or "")
        refined = data.get("query")
        return refined.strip() if isinstance(refined, str) and refined.strip() else query
    except Exception:
        logger.exception("질의 정제 실패, 원본 질의로 진행: %r", query)
        return query
