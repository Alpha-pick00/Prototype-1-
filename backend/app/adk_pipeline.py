"""ADK(Google Agent Development Kit) 기반 역할 분리형 검색 파이프라인.

검색(11번가) → 그라운딩(elevenst, 구조화·결정론적) → 제안(그라운딩 실패시만
조건부 DeepSeek 1개 + 소프트 교차확인) → 필터링+병합(fusion.dedup 재사용) →
검증(조건부 DeepSeek) → 매칭/합성 → 심사(조건부 Groq) 순서로 실행된다 —
`debate.py`의 run_single_debate/run_single_debate_stream이 이 모듈의
run()/run_stream()을 호출한다. (제안 슬롯 이름은 "gpt"/"groq" - "gpt" 슬롯은
실제로는 Qwen이 돌지만 리네임 비용이 커서 식별자를 그대로 뒀고, "groq" 슬롯은
2026-08-18에 실제 쓰는 모델명으로 리네임했다 - 원래 이름은 "gemini"였다.)

(2026-08-20) 정제(refine, Groq) 단계를 한 번 뺐다가("쿼리 재질의 없애고")
같은 날 다시 넣었다("'안녕 나 컵을 사고싶어'... 지금 LLM이 못 알아듣거든" -
대화체/인사말이 섞인 질의가 정제 없이 그대로 11번가 keyword 검색에 들어가면
검색·그라운딩이 둘 다 실패했다). 다만 예전처럼 매 질의마다 도는 게 아니라
looks_conversational_query()에 걸리는 질의에만 조건부로 돈다
(_skip_refine_unless_conversational) - 이미 짧고 깨끗한 검색어는 계속
건너뛴다. 검색 백엔드도 Tavily+다나와에서 11번가 오픈API로 바꿨다("다나와를
폐기하고 11번가 쪽으로") - 다나와 관련 코드(_DanawaFetchNode, fetchers/
danawa*.py 등)는 참고용으로 파일에 남아있지만 이 메인 파이프라인에서는 더
이상 안 쓴다.

(2026-08-20, "3개 LLM까지 필요없다") propose의 gpt(Qwen)/groq 슬롯도 뺐다 -
검색이 11번가 하나뿐이라 세 LLM이 elevenst 구조화 데이터와 완전히 같은
목록을 다시 텍스트로 추측해 읽고 있었다. 이제 elevenst 그라운딩이 성공하면
propose(deepseek 1개)·challenge(DeepSeek)·judge(Groq)·소프트 교차확인
(coupang/naver)까지 전부 조건부로 건너뛴다 - "행복 경로"(대부분의 검색)는
LLM 호출이 사실상 0번이고, elevenst 그라운딩이 실패했을 때만 deepseek가
의미적 매칭 안전망으로 조건부 호출된다.

SequentialAgent/ParallelAgent는 google-adk 2.6.3 기준 deprecated(대체 예정인
Workflow가 아직 LlmAgent의 sub-agent로 못 쓰여 미완성 상태)이지만, 실제로는
정상 동작함을 스파이크로 검증했다 — Workflow가 성숙하면 마이그레이션 대상.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, AsyncIterator
from urllib.parse import urlsplit

from fetchers import danawa as danawa_fetcher
from fetchers import elevenst as elevenst_fetcher
from openai import AsyncOpenAI
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.adk.models import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import TypeAdapter

from . import price_table as price_table_module
from . import search as search_module
from .category import classify_category
from .agents import deepseek as deepseek_module
from .agents import gpt as gpt_module
from .agents import judge as judge_module
from .intent import looks_conversational_query
from .agents.base import (
    NO_CANDIDATE_ERROR,
    build_challenge_prompt,
    build_prompt,
    build_refine_query_prompt,
    filter_candidates,
    format_results_block,
    is_danawa_comparison_page,
    parse_json_array,
    parse_json_object,
)
from .config import settings
from .schemas import (
    AgentCandidate,
    ChallengeResult,
    ChallengeVerdict,
    ClarifyResponse,
    DecideResponse,
    Decision,
    PriceTable,
    Proposal,
    RefinedQuery,
    SearchResult,
    StyleGuide,
    StyleGuideGroup,
)
from fusion.dedup import merge_candidates

logger = logging.getLogger(__name__)

_APP_NAME = "alpha_pick_debate"

# 매 요청 검색 히트 수 — search_cache.FETCH_SIZE(2026-08-19부터 20)와 동일하게.
# "이어폰"처럼 경쟁이 치열한 넓은 카테고리는 다나와 결과 상당수가 "가격비교
# 중지 상품"(구매 불가)이라 표본이 작으면 propose가 살아있는 후보를 하나도
# 못 볼 수 있다(search_cache.py의 FETCH_SIZE 상향 코멘트 참고) - 그 넓어진
# 표본을 실제로 다 쓰도록 맞춰 올린다.
_MAX_SEARCH_RESULTS = 20
# 에이전트당 최종 후보 1개(가장 좋은 것 하나)만 제안하게 한다(사용자 요청,
# 2026-08-15: "최종 후보도 각각 5개가 아닌 1개만 추천해주는걸로 하자 가장
# 좋은 거 1개") - 이전엔 에이전트당 최대 5개까지 브레인스토밍해 병합 풀을
# 넓혔지만(최대 15개), 그만큼 judge 심사 대상도 늘어나고 응답에도 서로 다른
# 후보가 여러 줄 노출될 수 있었다. 1로 줄이면 각 에이전트가 처음부터 자기
# 최선의 답 하나만 내고, 병합은 여전히 3개 에이전트가 같은 상품을 골랐는지
# 판단하는 데만 쓰인다.
_MAX_CANDIDATES_PER_AGENT = 1

# 취향 주도 카테고리(스타일 가이드 모드, 2026-08-19 사용자 요청)에 한해서만
# 위 1건 컷을 완화한다 - 위 2026-08-15 요청("1개만 추천")은 여전히 기본값
# (다른 모든 카테고리)에 그대로 적용되고, 이 값은 STYLE_GUIDE_CATEGORIES에
# 속한 질의에서만 쓰인다. propose 프롬프트가 이미 에이전트당 최대 5개를
# 배열로 돌려주므로(PROPOSAL_INSTRUCTIONS) 이 컷을 완화해도 새 LLM 호출은
# 없다 - 이미 받아온 응답을 덜 잘라낼 뿐이다.
_STYLE_GUIDE_MAX_CANDIDATES_PER_AGENT = 4

# judge/스타일 가이드 LLM 호출에 실제로 넣어주는 후보 수는 위 후보 풀 크기와
# 별개로 이 값에서 다시 한 번 자른다(2026-08-19 실측: 스타일 가이드 카테고리에서
# 후보 풀을 넓힌 채 이 컷 없이 judge 프롬프트를 만들면 이 프로젝트가 쓰는 Groq
# 계정의 gpt-oss-120b TPM 한도(분당 8000)를 넘겨 429(RateLimitError)가 났다 -
# "Request too large ... Limit 8000, Requested 8074/8283". judge는 원래도 "1개만
# 추천"(_MAX_CANDIDATES_PER_AGENT) 시절 최대 3개 안팎만 봤으므로, 이 컷은 그
# 예산 안으로 되돌리면서도 스타일 가이드가 2~4개 그룹을 만들 여유는 남긴다.
# state에 남는 proposals 자체(프론트 "다른 후보 보기"용)는 그대로 더 넓다 -
# LLM에게 "보여주는" 양만 줄인다.
_MAX_JUDGE_INPUT_CANDIDATES = 4

# category.py의 16개 대분류 중 "스펙보다 취향이 중요한" 카테고리만 스타일
# 가이드 대상으로 삼는다(사용자 요청, 2026-08-19: "다른 카테고리들도 저렇게
# 추가해줘" - 패션에서 시작해 확장). 나머지(식품/가전디지털/자동차용품 등)는
# 스펙·가격 비교가 핵심이라 지금처럼 단일 추천만 낸다. 필요하면 이 집합만
# 조정하면 된다.
STYLE_GUIDE_CATEGORIES = {"패션의류/잡화", "뷰티", "홈인테리어", "완구/취미"}


def _format_price_krw(price_krw: int | None) -> str:
    return f"{price_krw:,}원" if price_krw is not None else ""


def _search_results_from_state(state: dict) -> list[SearchResult]:
    return [SearchResult(**r) for r in state.get("search_results") or []]


def _refined_query_text(state: dict) -> str:
    refined = state.get("refined_query")
    if isinstance(refined, dict) and refined.get("query"):
        return refined["query"]
    return state.get("original_query", "")


# ---------------------------------------------------------------------------
# 커스텀(순수 Python) 노드 — ADK LlmAgent가 아니라 직접 상태를 읽고 쓴다.
# (_SearchNode는 예외적으로 내부에서 카테고리 분류를 한 번 호출한다 — 그 노드
# docstring 참고.)
# ---------------------------------------------------------------------------


_BROAD_FALLBACK_ELEVENST_LIMIT = 5


async def _broad_web_fallback_search(query: str) -> list[SearchResult]:
    """11번가 검색이 아무것도 못 찾았을 때의 최후 폴백(사용자 요청,
    2026-08-19: "검색 알고리즘으로 적절한 상품을 찾을 수 없는 경우에는 구글
    쇼핑에서 사용자 쿼리를 따로 검색해서 상위 5개의 제품을 가져오게").
    search_module.search_unrestricted()로 도메인 제한 없이 Tavily를 한 번 더
    호출해 질의에 맞는 실제 상품/브랜드명을 발견한 뒤, 가장 관련도 높은
    (Tavily가 1순위로 낸) 결과의 제목을 그 이름 삼아 11번가에 딱 한 번만
    재검색한다(최대 5개) - 다나와 배제(2026-08-20) 이후에도 이 최후 폴백은
    다나와 URL을 다시 만들어내지 않는다."""
    broad_results = await search_module.search_unrestricted(query)
    if not broad_results:
        return []

    discovered_name = broad_results[0].title
    if not settings.elevenst_api_key:
        return []
    try:
        result = await elevenst_fetcher.search_products(
            settings.elevenst_api_key, discovered_name, page_size=_BROAD_FALLBACK_ELEVENST_LIMIT
        )
    except elevenst_fetcher.ElevenstApiError:
        logger.warning("폴백 11번가 재검색 실패: %r", discovered_name, exc_info=True)
        return []

    return [
        SearchResult(
            title=p.name,
            url=p.detail_url,
            snippet=search_module._elevenst_snippet(p),
            score=None,
        )
        for p in result.products
        if p.detail_url and p.name
    ]


class _SearchNode(BaseAgent):
    """정제된 질의로 search_module.search()를 호출해 원본 결과 + 프롬프트용
    포맷 텍스트를 상태에 저장한다.

    한때(_augment_search_query, 2026-08-12) Tavily에 보내는 검색어 뒤에
    분류된 16개 대분류 카테고리 중 하나를 덧붙여 검색엔진 랭킹을 "미세
    조정"하려 했는데, 실측(2026-08-19 사용자 리포트: "10만원대 이어폰
    추천해줘 했는데 아무것도 안뜨잖아")으로 오히려 역효과임이 드러났다 -
    "가전디지털"처럼 원래 넓은 대분류를 덧붙이면 Tavily가 그 흔한 키워드
    쪽으로 결과를 완전히 끌고 가버려("이어폰" 검색이 스마트링·TV·자동차
    뉴스로 뒤덮임), "완만한 가중치"라는 설계 의도와 달리 원 질의 자체가
    묻혀버렸다. 그래서 이 보정을 제거하고 정제된 질의를 그대로 검색한다.
    classify_category는 이 노드의 유일한 호출부였다(clarify 단계는
    별도 호출, debate.py::_extract_clarify_options) - 이제 안 쓴다.

    (2026-08-20 갱신) search_module.search()는 다나와 대신 11번가 오픈API로
    검색한다("11번가 api를 구해서 다나와를 폐기하고 11번가 쪽으로 방향을
    틀려고") - 위 문단들의 "다나와/Tavily" 언급은 그 이전 구조에 대한
    역사적 맥락이다.

    11번가 검색이 완전히 빈손이면(2026-08-19 사용자 요청, 원래는 다나와
    한정 검색 대상) _broad_web_fallback_search로 한 번 더 시도한다 - Tavily
    비제한 검색으로 상품/브랜드명을 발견한 뒤 11번가에 재검색하는 최후
    수단이라, 대부분의 검색은 이 분기를 안 탄다.

    classify_category는(2026-08-19, 스타일 가이드 모드 게이트로 재사용 -
    검색어 보정 용도로는 위에서 이미 뗐지만 이건 별개 목적이다) 검색과
    asyncio.gather로 나란히 돌린다 - Tavily 검색이 어차피 훨씬 오래 걸려서
    같이 부르면 이 카테고리 분류 지연시간이 전혀 추가되지 않는다."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        query = _refined_query_text(ctx.session.state)
        search_task = search_module.search(query, max_results=_MAX_SEARCH_RESULTS)
        classify_task = classify_category(query, [])
        results, classification = await asyncio.gather(
            search_task, classify_task, return_exceptions=True
        )
        if isinstance(results, BaseException):
            logger.exception("검색 실패: %r", query, exc_info=results)
            results = []
        if isinstance(classification, BaseException):
            logger.exception("스타일 가이드 카테고리 분류 실패: %r", query, exc_info=classification)
            style_guide_mode = False
        else:
            style_guide_mode = classification.category in STYLE_GUIDE_CATEGORIES

        if not results:
            try:
                results = await _broad_web_fallback_search(query)
            except Exception:
                logger.exception("비제한 폴백 검색 실패: %r", query)
                results = []

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "search_results": [r.model_dump() for r in results],
                    "search_results_block": format_results_block(results),
                    "style_guide_mode": style_guide_mode,
                }
            ),
        )


class _DanawaFetchNode(BaseAgent):
    """다나와 실측 가격표를 propose 3개 모델과 나란히(동시에) 조회한다 —
    PRESERVED FROM seungmin/lsm의 run_single_debate_price_table_variant(PART 4-2)를
    ADK 파이프라인으로 포팅(2026-08-16, README "한계점 및 향후 과제" 후속작업).
    같은 ParallelAgent(propose_parallel) 소속이라 gpt/groq/deepseek LlmAgent와
    동시에 실행되므로 지연시간이 추가되지 않는다.

    price_table_module.fetch_price_tables()는 원래도 예외를 던지지 않지만,
    이 노드도 다른 커스텀 노드(_SearchNode 등)와 같은 방어 패턴을 따라 한 번 더
    감싼다 — 다나와 조회가 실패해도 gpt/groq/deepseek 후보만으로 파이프라인이
    계속 진행된다.

    A등급(구매 링크 생성 가능) 최저가 offer가 있는 대표 가격표(pick_primary -
    offer가 가장 많은 페이지) 하나만 골라 다른 슬롯과 같은 모양(raw JSON
    문자열, output_key 컨벤션)으로 danawa_raw에 저장한다 - _FilterMergeNode가
    다른 3개 슬롯과 동일하게 처리해 병합 풀에 합류시킨다. 여러 다나와 페이지가
    있어도 대표 1개만 후보로 올리는 이유: 다른 3개 에이전트도 이미 "각자 최선
    1개"만 제안하도록 되어 있어(_MAX_CANDIDATES_PER_AGENT) 여러 개를 넣어도
    filter_candidates가 어차피 1개로 자른다.

    danawa_tables(전체 가격표, 모든 offer 포함)는 별도로 저장해 run_stream()의
    후처리(enrich_decision/exclude_price_comparison_site_as_final_pick/
    price_table 채우기)에서 그대로 재사용한다 - 이 노드에서 만든 대표 후보
    하나로는 그 후처리들이 필요로 하는 전체 가격표 정보(모든 offer, 다른 pcode
    페이지들)를 담을 수 없다.

    (2026-08-16 하드닝, 그라운딩 회귀 파일럿에서 발견) pick_primary()는
    offer가 가장 많은 페이지를 고를 뿐 query와의 관련성은 전혀 안 본다 -
    "아이폰 16 프로 256GB" 검색에서 실제로 무관한 액세서리(아이패드 케이스,
    판매처가 많아 offer 수만 높음)가 골라져 verified=True로 강제된 채(아래
    _apply_challenge 참고 - danawa 출신 후보는 challenge 검증 자체를 건너뜀)
    최종 추천으로 노출된 사례를 확인했다. enrich_decision이 이미 쓰는
    _product_name_matches(fuzzy 유사도 + 모델/수량 충돌 가드)를 query 자체에도
    적용해, 이름이 안 맞으면 애초에 후보를 만들지 않는다 - danawa_tables(전체
    가격표)는 그대로 보존해 다른 후처리는 영향받지 않는다."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        query = _refined_query_text(ctx.session.state)
        results = _search_results_from_state(ctx.session.state)

        tables_dump: list[list] = []
        danawa_raw = "[]"
        try:
            tables = await price_table_module.fetch_price_tables(query, results)
            tables_dump = [[table.model_dump(), raw] for table, raw in tables]
            primary = price_table_module.pick_primary(tables)
            if primary is not None:
                primary_table, raw_result = primary
                if price_table_module._product_name_matches(query, primary_table.product_name or ""):
                    offer = price_table_module.cheapest_linkable_raw_offer(raw_result)
                    if offer is not None:
                        resolved_url = await price_table_module.resolve_purchase_url(offer)
                        if resolved_url is not None:
                            candidate = AgentCandidate(
                                product_name=primary_table.product_name or query,
                                price_krw=offer["price_krw"],
                                retailer=offer["seller"],
                                url=resolved_url,
                            )
                            danawa_raw = json.dumps([candidate.model_dump()])
        except Exception:
            logger.exception("다나와 실측가 조회 실패: %r", query)

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={"danawa_tables": tables_dump, "danawa_raw": danawa_raw}
            ),
        )


def _pick_elevenst_candidate(query: str, result: elevenst_fetcher.SearchResult) -> AgentCandidate | None:
    """순수 함수(네트워크 없음) - _ElevenstFetchNode의 후보 선정 로직을
    분리해 ADK 배관 없이 단위 테스트할 수 있게 한다.

    다나와의 pick_primary()처럼 "가장 풍부한 페이지" 개념이 없다 -
    ProductSearch는 개별 SKU 목록이라, query와 그라운딩되는(price_table
    module._product_name_matches) 상품 중 최저가 1건을 고른다(_DanawaFetchNode가
    query 무관 상품을 걸러낸 2026-08-16 하드닝과 같은 이유 - "이프로"를 검색해서
    전혀 다른 상품이 최저가라는 이유만으로 뽑히면 안 된다). 유효 가격
    (sale_price 우선, 없으면 price)과 detail_url이 둘 다 있어야 후보가 된다 -
    구매 링크 없는 추천은 만들지 않는다."""
    matched = [
        p
        for p in result.products
        if p.name and price_table_module._product_name_matches(query, p.name)
    ]
    priced = [(p, p.sale_price or p.price) for p in matched]
    priced = [(p, price) for p, price in priced if price and price > 0 and p.detail_url]
    if not priced:
        return None
    cheapest, price = min(priced, key=lambda pair: pair[1])
    return AgentCandidate(
        product_name=cheapest.name,
        price_krw=price,
        retailer="11번가",
        url=cheapest.detail_url,
    )


def _elevenst_grounded(state: dict) -> bool:
    """이 시점(_ElevenstFetchNode 완료 후)에 11번가 그라운딩이 이미 성공했는지 -
    propose 단계의 LLM 폴백/소프트 교차확인 노드들이 자신을 건너뛸지 판단하는
    데 공유로 쓴다(2026-08-20, "3개 LLM까지 필요없다" 조건부 게이팅 참고,
    _ElevenstFetchNode·_skip_propose_if_elevenst_grounded·_CoupangCheckNode·
    _NaverCheckNode)."""
    try:
        return bool(json.loads(state.get("elevenst_raw") or "[]"))
    except (TypeError, ValueError):
        return False


class _ElevenstFetchNode(BaseAgent):
    """11번가 오픈API(ProductSearch)로 구조화 후보를 조회한다(2026-08-20,
    "11번가 api를 구해서 다나와를 폐기하고 11번가 쪽으로 방향을 틀려고").

    (2026-08-20 재구성, "3개 LLM까지 필요없다") 원래는 propose_parallel 소속으로
    gpt/groq/deepseek LlmAgent와 동시 실행됐는데, 이제 propose_parallel보다
    앞선 별도 순차 단계로 옮겼다 - propose의 유일한 LLM(deepseek, 폴백 전용)과
    coupang_check/naver_check가 "elevenst 그라운딩이 이미 성공했는지"를 보고
    자신을 건너뛸지 정해야 하는데, 병렬로 동시 실행되면 그 판단 시점에 이
    노드가 아직 안 끝났을 수 있어(순서 보장이 없음) 판단 자체가 불가능했다.
    순차 단계로 옮기면 SequentialAgent가 이 노드의 완료(state_delta 반영)를
    보장한 뒤에야 propose_parallel이 시작되므로, 뒤이은 노드들이 _elevenst_
    grounded(state)를 신뢰성 있게 볼 수 있다. 대가는 11번가 API 호출 하나가
    이제 LLM 호출과 겹치지 않고 순차로 들어간다는 것(수백 ms) - 그라운딩이
    성공하는 대부분의 경우 그 대가로 LLM 호출 자체(들)를 통째로 생략하므로
    순이익이 훨씬 크다.

    elevenst_fetcher.search_products()는 예외를 던질 수 있어(11번가 쿼터/장애
    거동이 검증 안 됨) 이 노드가 직접 감싼다 - 실패해도(elevenst_raw="[]")
    뒤이은 deepseek 폴백이 조건부로 살아나 파이프라인이 계속 진행된다. API 키
    미설정 시에도 조용히 스킵한다(ocr/google_vision.py와 동일한 패턴)."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        query = _refined_query_text(ctx.session.state)

        elevenst_raw = "[]"
        if settings.elevenst_api_key:
            try:
                result = await elevenst_fetcher.search_products(settings.elevenst_api_key, query)
                candidate = _pick_elevenst_candidate(query, result)
                if candidate is not None:
                    elevenst_raw = json.dumps([candidate.model_dump()])
            except Exception:
                logger.exception("11번가 검색 실패: %r", query)

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"elevenst_raw": elevenst_raw}),
        )


class _CoupangCheckNode(BaseAgent):
    """challenge 단계에 쿠팡 검색 결과를 독립 교차 확인 신호로 추가한다
    (사용자 요청, 2026-08-16: "그라운딩 성능을 높여줘"). propose_parallel
    소속이라 (조건부로 살아나는) deepseek 폴백과 동시 실행되어 지연시간이
    추가되지 않는다. 다나와처럼 가격을 추출해 후보로 올리지는 않는다 - 쿠팡
    페이지를 직접 파싱하지 않고(과거 15개 리테일러를 다나와 하나로 좁힌
    이유였던 "스니펫만으로 파싱하면 엉뚱한 상품/가격이 섞이는 문제"를
    재현하지 않기 위해) Tavily 스니펫만 challenge LLM에게 참고 신호로 넘긴다.

    (2026-08-20, "3개 LLM까지 필요없다" 비용 점검에서 같이 발견) elevenst
    그라운딩이 이미 성공했으면 이 신호는 아무도 안 읽는다 - _apply_challenge가
    구조화 소스 후보의 challenge verdict를 애초에 무시하고, challenge LLM
    호출 자체도 _skip_challenge_if_all_structured가 건너뛰므로 이 검색
    결과를 담을 프롬프트가 아예 안 만들어진다. 그래서 elevenst가 그라운딩에
    성공한 경우엔 이 Tavily 호출도 같이 건너뛴다."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        if _elevenst_grounded(state):
            yield Event(author=self.name, actions=EventActions(state_delta={"coupang_results": []}))
            return
        query = _refined_query_text(state)
        results = await search_module.search_coupang(query)
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"coupang_results": [r.model_dump() for r in results]}),
        )


class _NaverCheckNode(BaseAgent):
    """쿠팡(_CoupangCheckNode)과 동일한 패턴의 두 번째 소프트 그라운딩 신호
    (2026-08-16, "다나와 단일 실측 소스에 대한 의존도를 낮추도록") - 다나와
    실측가만 확정 소스이고 쿠팡 하나만으로는 교차 확인 대상이 한 곳뿐이라,
    서로 다른 두 번째 쇼핑몰을 propose_parallel에 같이 태워 challenge가 볼
    참고 자료를 넓힌다. 후보를 만들지 않는 점, 실패해도 조용히 빈 리스트인
    점, elevenst 그라운딩 성공 시 건너뛰는 점 모두 쿠팡 노드와 동일(이유는
    _CoupangCheckNode 참고)."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        if _elevenst_grounded(state):
            yield Event(author=self.name, actions=EventActions(state_delta={"naver_results": []}))
            return
        query = _refined_query_text(state)
        results = await search_module.search_naver(query)
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"naver_results": [r.model_dump() for r in results]}),
        )


class _FilterMergeNode(BaseAgent):
    """제안 노드(elevenst + 조건부 deepseek 폴백) + 다나와 후보의 원시 JSON을
    각각 파싱+필터링한 뒤, fusion.dedup으로 동일 상품을 병합한다 — 지금까지
    어디서도 안 쓰이던 merge_candidates()가 여기서 처음 실사용된다.

    (2026-08-20, "3개 LLM까지 필요없다") gpt/groq 슬롯은 propose_parallel에서
    빠져서 raw_by_agent에도 없다 - _build_propose_agent 자체는 남아있으므로
    필요하면 다시 추가할 수 있지만, 지금은 elevenst(구조화, 항상 시도) +
    deepseek(그라운딩 실패시에만 조건부 호출) 둘뿐이다."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        raw_by_agent = {
            "deepseek": state.get("deepseek_raw"),
            # "danawa": 2026-08-20부터 propose_parallel에서 빠져서 항상 None(무해) -
            # 다나와 관련 코드/데이터 흐름은 그대로 남겨뒀으므로 이 키 자체는 지우지
            # 않는다. "elevenst"가 메인 파이프라인의 구조화 후보 슬롯을 대신한다.
            "danawa": state.get("danawa_raw"),
            "elevenst": state.get("elevenst_raw"),
        }
        danawa_tables = _danawa_tables_from_state(state)
        # 취향 주도 카테고리(스타일 가이드 모드)는 에이전트당 후보를 1개로
        # 자르지 않는다 - propose 프롬프트가 이미 매번 최대 5개를 배열로
        # 돌려주므로(PROPOSAL_INSTRUCTIONS) 새 LLM 호출 없이 표본만 넓어진다.
        max_per_agent = (
            _STYLE_GUIDE_MAX_CANDIDATES_PER_AGENT
            if state.get("style_guide_mode")
            else _MAX_CANDIDATES_PER_AGENT
        )
        merged = await _merge_proposals(raw_by_agent, danawa_tables, max_candidates_per_agent=max_per_agent)
        logger.info(
            "후보 풀: 병합 %d건 (%r)", len(merged), [m["proposed_by"] for m in merged]
        )

        yield Event(author=self.name, actions=EventActions(state_delta={"candidates": merged}))


async def _resolve_comparison_page_item(
    item: dict, danawa_tables: list[tuple[PriceTable, dict]]
) -> dict:
    """item['url']이 다나와 가격비교 페이지(prod.danawa.com/info?pcode=...)면
    같은 propose 라운드에서 _DanawaFetchNode가 이미 페치해 둔 가격표와 pcode로
    대조해 A등급 최저가 구매링크로 바꿔치기한다 - 해석에 성공하면 URL이
    /bridge/ 형태로 바뀌어 뒤이은 filter_candidates()의
    is_danawa_comparison_page 검사를 정상 통과한다. 해석 실패(pcode 불일치,
    A등급 오퍼 없음 등)하면 원본을 그대로 반환해 기존처럼 필터링되게 둔다 -
    안 검증된 값을 지어내지 않는다는 원칙은 그대로 유지."""
    url = item.get("url") or ""
    if not is_danawa_comparison_page(url):
        return item
    resolved = await price_table_module.resolve_danawa_comparison_url(url, danawa_tables)
    if resolved is None:
        return item
    resolved_url, price_krw, retailer = resolved
    return {**item, "url": resolved_url, "price_krw": price_krw, "retailer": retailer}


async def _merge_proposals(
    raw_by_agent: dict[str, str | None],
    danawa_tables: list[tuple[PriceTable, dict]],
    max_candidates_per_agent: int = _MAX_CANDIDATES_PER_AGENT,
) -> list[dict]:
    """3개 제안자의 원시 JSON 텍스트를 각각 파싱한 뒤, 다나와 가격비교 페이지
    URL은 A등급 구매링크로 먼저 해석하고(_resolve_comparison_page_item),
    필터링해 fusion.dedup으로 동일 상품을 병합한다 — 한 제안자의 파싱이
    실패해도 나머지로 진행한다.

    (2026-08-18) 원래는 LLM 호출 없이 테스트 가능한 순수 동기 함수였는데,
    Qwen/Groq/DeepSeek이 다나와 검색 결과에서 고를 수 있는 URL이 사실상 전부
    가격비교 페이지 형태뿐이라(다나와 도메인 단독 검색의 특성상) 해석 없이
    그대로 필터링하면 세 제안자가 후보를 거의 못 만들어 후보 풀이 자주
    0건으로 비고, 그 결과 되묻기(clarify)나 "적절한 상품 후보를 찾지 못했다"
    실패가 대량 발생하는 걸 그라운딩 회귀 파일럿에서 확인했다(50개 중 정직한
    실패인 케이스는 12%뿐, 나머지는 이 문제였다). 해석 자체(resolve_purchase_url)는
    실제로는 네트워크 호출이 없는 순수 문자열 가공이라(price_table.py 참고)
    async로 바꿔도 테스트 비용이 늘지 않는다."""
    entries: list[tuple[str, AgentCandidate]] = []
    for agent_name, raw in raw_by_agent.items():
        if not raw:
            continue
        try:
            items = parse_json_array(raw)
            items = [await _resolve_comparison_page_item(item, danawa_tables) for item in items]
            items = filter_candidates(items, max_items=max_candidates_per_agent)
            entries.extend((agent_name, AgentCandidate(**item)) for item in items)
        except Exception:
            logger.exception("%s 제안 파싱 실패, 이 제안자는 후보 풀에서 제외", agent_name)

    return merge_candidates(entries)


# Tavily extract 호출 수 상한 — 병합 후보가 많아도 재조회 비용/지연시간을 제한한다.
_MAX_EXTRACT_CANDIDATES = 10

# challenge 검증 없이도 이미 구조화·검증된 소스로 취급하는 에이전트 이름 집합
# (2026-08-20, "다나와를 폐기하고 11번가 쪽으로") - _apply_challenge와
# _urls_needing_challenge_extract가 공유한다. "danawa"도 계속 포함해두는 이유:
# _DanawaFetchNode 자체는 코드로 남겨뒀고(propose_parallel에서만 빠짐),
# run_danawa_only_debate() 같은 별도 경로가 여전히 danawa_raw 규약을 쓸 수
# 있다 - 이 집합에 남겨두는 게 가장 안전하고 최소 변경이다.
STRUCTURED_SOURCE_AGENTS = {"danawa", "elevenst"}

# _apply_challenge()가 challenge_note를 조립할 때 쓰는 소스별 표시 문구.
_STRUCTURED_SOURCE_NOTE = {
    "danawa": "다나와 실측 가격표(A등급, 구매 링크 검증됨)",
    "elevenst": "11번가 공식 API",
}


def _urls_to_extract(candidates: list[dict]) -> list[str]:
    """재조회 대상 URL — 병합 후보 중 URL이 있는 것만, 상한선까지만 남긴다."""
    return [c["url"] for c in candidates[:_MAX_EXTRACT_CANDIDATES] if c.get("url")]


def _urls_needing_challenge_extract(candidates: list[dict]) -> list[str]:
    """Tavily extract() 결과가 실제로 challenge 판정에 쓰이는 URL만 남긴다
    (2026-08-20, 성능/비용 점검 - "다나와 후보 extract() 낭비 없애줘").

    _apply_challenge()는 proposed_by에 STRUCTURED_SOURCE_AGENTS(다나와/11번가)가
    있는 후보는 challenge 검증 결과를 아예 안 본다 - verified를 무조건 True로
    강제하고, refreshed_price_krw도 절대 안 쓴다(공식/실측 구조화 데이터가
    텍스트 재추출보다 정확하다고 보기 때문). 그 URL의 extract() 결과는 challenge
    프롬프트에 실리기만 하고 최종적으로 어디에도 반영되지 않으므로, 애초에
    Tavily에 요청할 필요가 없다.

    URL 도메인이 아니라 proposed_by로 판단하는 이유 - propose 에이전트(Qwen/
    Groq/DeepSeek)가 검색 스니펫만 보고 독자적으로 고른 후보도 URL이 우연히
    11번가/다나와 도메인일 수 있다. 그런 후보는 구조화 소스 노드의 픽과
    병합되지 않았다면(proposed_by에 없다면) challenge 검증이 그대로 쓰이므로,
    도메인 기준으로 걸렀다면 진짜 필요한 재조회까지 함께 잘려나갔을 것이다."""
    return [
        c["url"]
        for c in candidates[:_MAX_EXTRACT_CANDIDATES]
        if c.get("url") and not (STRUCTURED_SOURCE_AGENTS & set(c.get("proposed_by") or []))
    ]


def _is_danawa_product_url(url: str | None) -> bool:
    return bool(url) and urlsplit(url).netloc.lower().endswith("danawa.com")


class _ExtractPagesNode(BaseAgent):
    """병합된 후보들의 실제 판매 페이지 원문을 다시 가져와 상태에 저장한다.
    challenge(DeepSeek)는 기존에는 검색 당시 잘린 스니펫(최대 1500자)만 보고
    판단했는데, 여기서 채운 candidate_pages를 함께 주면 지금 이 URL의 실제
    최신 본문을 근거로 재검증할 수 있다 — search.extract()는 원래도 있었지만
    어디서도 쓰이지 않던 함수였다. 개별 URL 조회가 실패해도 그 후보만 스니펫
    기반 검증으로 남고 나머지는 그대로 진행한다(전체를 막지 않음).

    후보 URL은 전부 다나와 검색(Tavily include_domains=danawa.com)에서 나오므로,
    다나와 URL에 한해 fetchers/danawa.py로 직접 재조회해 "가격비교 서비스가
    종료된 상품" 페이지인지도 함께 확인한다(사용자 요청, 2026-08-13: "가격미확인
    되어있는것도 뜨거든... 가격 비교가 중지된 상품이라고 뜨는 이런 경우의 수도
    없애주면"). Tavily extract 텍스트만으로 challenge LLM이 이 패턴을 매번
    알아채리라 보장할 수 없어서, _is_expired_page()로 이미 검증된 판별을
    LLM 판단에 맡기지 않고 여기서 결정적으로 걸러낸다.

    Tavily extract()는 _urls_needing_challenge_extract()로 거른 URL에만 친다
    (2026-08-20) - proposed_by에 "danawa"가 있는 후보는 _apply_challenge가
    challenge 검증 결과를 아예 안 쓰므로(다나와 자체 스크래핑 가격을 그대로
    신뢰), 그 URL까지 재조회하는 건 순수 낭비였다. 만료 여부 확인
    (_check_expired)은 이 구분과 무관하게 danawa.com 도메인이면 전부 그대로
    돌린다 - "가격비교 중지" 여부는 challenge 검증과 별개로 모든 다나와 후보에
    똑같이 적용돼야 하는 사실 확인이라 대상을 줄이면 안 된다."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        candidates: list[dict] = ctx.session.state.get("candidates") or []
        urls = _urls_to_extract(candidates)
        extract_urls = _urls_needing_challenge_extract(candidates)
        danawa_urls = [u for u in urls if _is_danawa_product_url(u)]

        async def _safe_extract(url: str) -> tuple[str, str | None]:
            try:
                return url, await search_module.extract(url)
            except Exception:
                logger.warning("후보 페이지 재조회 실패: %r", url, exc_info=True)
                return url, None

        async def _check_expired(url: str) -> tuple[str, bool]:
            try:
                result = await danawa_fetcher.fetch_danawa_offers(url)
                return url, result["parse_status"] == "expired"
            except Exception:
                logger.warning("다나와 후보 URL 상태 확인 실패: %r", url, exc_info=True)
                return url, False

        pages: dict[str, str] = {}
        expired_urls: list[str] = []
        if urls:
            extracted, expiry_checked = await asyncio.gather(
                asyncio.gather(*(_safe_extract(u) for u in extract_urls)),
                asyncio.gather(*(_check_expired(u) for u in danawa_urls)),
            )
            pages = {url: text for url, text in extracted if text}
            expired_urls = [url for url, is_expired in expiry_checked if is_expired]

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"candidate_pages": pages, "expired_danawa_urls": expired_urls}),
        )


class _ApplyChallengeNode(BaseAgent):
    """DeepSeek의 검증 결과를 병합된 후보와 매칭해 최종 Proposal 목록을 만든다 —
    URL 우선, 실패 시 인덱스로 폴백(LLM이 순서를 흐트러뜨릴 가능성에 대비)."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        candidates: list[dict] = state.get("candidates") or []
        raw_challenge = state.get("raw_challenge")
        expired_urls = set(state.get("expired_danawa_urls") or [])

        verdicts: list[ChallengeVerdict] = []
        if raw_challenge:
            try:
                verdicts = [ChallengeVerdict(**v) for v in parse_json_array(raw_challenge)]
            except Exception:
                logger.exception("DeepSeek 검증 결과 파싱 실패 — 전부 미검증으로 진행")

        proposals = _apply_challenge(candidates, ChallengeResult(verdicts=verdicts), expired_urls)

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"proposals": [p.model_dump() for p in proposals]}),
        )


def _apply_challenge(
    candidates: list[dict], challenge: ChallengeResult, expired_urls: set[str] = frozenset()
) -> list[Proposal]:
    """병합된 후보(merge_candidates 출력 dict)와 검증 결과를 매칭해 Proposal로
    합성하는 순수 함수 — LLM 호출 없이 테스트 가능.

    expired_urls에 속한 후보(다나와가 "가격비교 서비스가 종료된 상품"으로 표시하는
    페이지로 확인됨)는 verified=False로 남기지 않고 아예 결과에서 제외한다 —
    challenge/judge 단계까지 흘려보내면 죽은 링크가 "가격미확인" 카드로 그대로
    노출될 수 있어서다. 후보가 전부 걸러지면 proposals가 빈 리스트가 되는데,
    이는 run_stream()의 기존 "후보 0개" 폴백 체인(clarify → relaxed fallback →
    NO_CANDIDATE_ERROR)이 그대로 처리한다.

    proposed_by에 "danawa"가 섞인 후보(_DanawaFetchNode가 넣은 실측가 - 다른
    에이전트와 병합됐든 단독이든)는 DeepSeek 검증 결과를 안 보고 verified=True로
    강제한다 - 이미 실측 확인된 가격/구매링크를 텍스트 기반 challenge LLM에게
    다시 판단하게 하는 건 불필요하고, verdict가 안 붙으면 verified=None(미검증)
    으로 표시돼 실측 데이터가 오히려 검증 안 된 것처럼 보이는 문제도 막는다.

    가격 신선도(2026-08-19, 사용자 리포트 "실제로 사이트 들어갔을 때는 다른
    가격을 가져오는 문제") - danawa 출신이 아닌 후보(주로 propose 단계 LLM이
    검색 캐시 스니펫을 읽고 추측한 price_krw, 최대 6시간 묵을 수 있음)는
    verdict.refreshed_price_krw가 있으면 그걸로 price를 덮어쓴다. 이 값은 새
    네트워크 호출이 아니라 challenge 직전 _ExtractPagesNode가 이미 라이브로
    재조회해 둔 candidate_pages 원문에서 나온다 - 그동안 그라운딩 검증에만
    쓰이고 버려지던 데이터를 가격 갱신에도 재사용하는 것. danawa 출신 후보는
    원래 가격 자체가 이미 구조화된 실측 스크래핑이라(대신 최대 1시간 캐시)
    텍스트 기반 재추출이 오히려 덜 정확할 수 있어 건드리지 않는다."""
    verdicts_by_url = {v.url: v for v in challenge.verdicts if v.url}

    proposals = []
    for i, candidate in enumerate(candidates):
        url = candidate.get("url")
        if url in expired_urls:
            logger.info("다나와 가격비교 중지 페이지로 확인되어 후보에서 제외: %s", url)
            continue

        proposed_by = candidate.get("proposed_by") or []
        price_krw = candidate.get("price_krw")
        structured_sources = STRUCTURED_SOURCE_AGENTS & set(proposed_by)
        if structured_sources:
            verified: bool | None = True
            note_labels = [_STRUCTURED_SOURCE_NOTE[s] for s in ("danawa", "elevenst") if s in structured_sources]
            challenge_note = " + ".join(note_labels) + "에서 확인된 후보"
        else:
            verdict = verdicts_by_url.get(url)
            if verdict is None and i < len(challenge.verdicts):
                verdict = challenge.verdicts[i]
            verified = verdict.verified if verdict else None
            challenge_note = verdict.note if verdict else None
            if verdict is not None and verdict.refreshed_price_krw is not None:
                price_krw = verdict.refreshed_price_krw
                challenge_note = (
                    f"{challenge_note} (재조회 시점 가격으로 갱신됨)"
                    if challenge_note
                    else "재조회 시점 가격으로 갱신됨"
                )

        # 같은 상품이 여러 모델에서 조금씩 다른 문구로 제안되면 reasons가 여러 개
        # 쌓인다 — 전부 이어붙이면 근거가 아니라 벽글이 되므로 최대 2개만 보여준다.
        reasons = (candidate.get("reasons") or [])[:2]
        proposals.append(
            Proposal(
                agent=proposed_by[0] if proposed_by else "gpt",
                product_name=candidate.get("product_name"),
                price=_format_price_krw(price_krw),
                retailer=candidate.get("retailer"),
                url=candidate.get("url"),
                reasoning=" / ".join(reasons) if reasons else None,
                verified=verified,
                challenge_note=challenge_note,
                proposed_by=proposed_by or None,
            )
        )
    return proposals


# ---------------------------------------------------------------------------
# LlmAgent 노드 — 실제 모델 호출은 ADK + LiteLlm이 담당(수동 SDK 호출 없음).
#
# ADK의 SequentialAgent/ParallelAgent는 서브 에이전트 하나가 예외를 던지면
# 파이프라인 전체를 그대로 죽인다 - 이 프로젝트는 이 기본 동작을 그대로
# 쓴다(2026-08-18, 사용자 요청: "AI 모델중에 하나라도 토큰 다쓰면 실행되지
# 않도록 바꿔줘" - "3개중 하나라도 빠지면 결과를 내지 않도록"과 같은 원칙을
# refine/challenge/judge까지 전부 확장). challenge·judge는 on_model_error_callback을
# 아예 등록하지 않아 원래도 실패하면 그대로 예외가 올라갔다(agent.
# canonical_on_model_error_callbacks가 비어있으면 ADK가 원본 예외를 그대로
# raise한다 - base_llm_flow._call_llm_async 참고). refine/propose 2곳만
# "이 모델 하나 없이도 계속 진행"하는 콜백을 등록해뒀었는데(과거엔 모델 하나가
# 죽어도 나머지로 답을 냈다), 이제 둘 다 실패를 그대로 흘려보내 pipeline_failed로
# 처리되게 바꿨다 - run_stream()의 바깥 try/except가 이를 잡아 proposals를
# 빈 채로 두고 기존 clarify/relaxed fallback/NO_CANDIDATE_ERROR 경로로 이어진다.
# ---------------------------------------------------------------------------


def _model_error_fallback_response(text: str) -> LlmResponse:
    """모델 호출이 실패했을 때 성공한 것처럼 대신 흘려보낼 최소 응답 — ADK는 이
    텍스트를 실제 모델 응답과 동일하게 output_key/output_schema 처리 경로로
    흘려보낸다(base_llm_flow._finalize_model_response_event가 이 content를 그대로
    이벤트에 병합)."""
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _skip_refine_unless_conversational(callback_context, llm_request) -> LlmResponse | None:
    """정제(refine)는 2026-08-20("쿼리 재질의 없애고")에 한 번 파이프라인에서
    완전히 빠졌었다 - "안녕 나 컵을 사고싶어"처럼 인사말/대화체로 감싼 질의가
    정제 없이 그대로 11번가 API keyword로 들어가면서 검색도 그라운딩도 실패하는
    회귀가 드러나(같은 날 사용자 리포트) 다시 넣었다. 다만 예전처럼
    needs_clarification() 전체(짧고 깨끗한 검색어 "음료수"까지 포함)를 기준으로
    삼지 않는다 - 그건 정제가 필요 없는 대부분의 짧은 검색어까지 매번 LLM
    왕복을 태워, 이번 세션에서 줄인 LLM 호출 수를 도로 늘린다. 대신
    looks_conversational_query()로 범위를 좁혀, 정말 "인사말/대화체 문장에 진짜
    검색어가 섞여 있는" 경우에만 정제를 태운다 - 그 외(이미 짧고 깨끗한 검색어,
    이미 구체적인 검색어)는 건너뛴다."""
    original_query = callback_context.state.get("original_query", "")
    if not original_query or looks_conversational_query(original_query):
        return None
    fallback = RefinedQuery(query=original_query)
    return _model_error_fallback_response(fallback.model_dump_json())


def _on_refine_model_error(callback_context, llm_request, error) -> LlmResponse | None:
    """refine의 모델(Groq) 호출이 실패하면 그 실패를 그대로 흘려보낸다(None
    반환) - propose와 동일한 이유(_on_propose_model_error 참고). 원래는
    정제를 포기하고 원본 질의로 폴백해 계속 진행했는데(다듬지 않은
    질의로라도 검색을 계속하는 게 파이프라인을 죽이는 것보다 낫다는 판단),
    2026-08-18(사용자 요청: "AI 모델중에 하나라도 토큰 다쓰면 실행되지
    않도록 바꿔줘") 어떤 모델이든 하나라도 실패하면 정직하게 전체를
    실패시키는 쪽으로 원칙을 통일했다."""
    logger.warning(
        "refine 단계 모델 호출 실패 - 모델 하나라도 실패하면 파이프라인을 그대로 실패시킨다",
        exc_info=error,
    )
    return None


def _on_propose_model_error(callback_context, llm_request, error) -> LlmResponse | None:
    """propose LLM 호출이 실패하면 그 실패를 그대로 흘려보낸다(None 반환) -
    ADK는 이 콜백이 None을 반환하면 원래 예외를 그대로 raise한다
    (base_llm_flow._call_llm_async 참고). 그 예외는 run_stream()의 바깥
    try/except가 잡아 pipeline_failed=True로 처리하고, proposals가 빈 채로
    기존 clarify/relaxed fallback/NO_CANDIDATE_ERROR 경로로 이어진다.

    (2026-08-18, 사용자 요청: "3개중 하나라도 빠지면 결과를 내지 않도록
    해야지 왜 3개를 안쓰고 2개만해서 결과를 내") 원래는 실패한 슬롯만 빈
    배열("[]")로 대체해 나머지로 계속 진행했다 - 여러 모델이 독립적으로
    같은 상품을 골랐는지(proposed_by 합의)가 그라운딩 신뢰도 신호로 쓰이는
    구조라, 일부만으로 낸 답은 원래 설계한 신뢰도보다 낮은데도 겉보기엔
    구분이 안 됐다. 하나라도 빠지면 아예 정직하게 실패로 처리하기로 했다.

    (2026-08-20, "3개 LLM까지 필요없다") propose LLM 슬롯이 deepseek 하나만
    남아 이 콜백을 쓰는 건 이제 그 하나뿐이지만, 원칙은 그대로 유지한다 -
    elevenst 그라운딩이 이미 성공했으면 이 LLM은 애초에 호출되지 않고
    (_skip_propose_if_elevenst_grounded), 실패해서 정말 이 콜백까지 왔다는
    건 elevenst도 실패했고 유일한 폴백마저 실패했다는 뜻이라 정직하게
    실패시키는 게 더더욱 맞다."""
    logger.warning(
        "%s propose 단계 모델 호출 실패 - 파이프라인을 그대로 실패시킨다",
        callback_context.agent_name,
        exc_info=error,
    )
    return None


def _skip_propose_if_elevenst_grounded(callback_context, llm_request) -> LlmResponse | None:
    """propose의 유일한 LLM(deepseek)을 elevenst 그라운딩이 이미 성공했을 때
    건너뛴다(2026-08-20, "3개 LLM까지 필요없다" - 사용자가 3개는 과하다고
    판단, 그렇다고 0개면 rapidfuzz 그라운딩(_product_name_matches)이 못 잡는
    오타/비속어/다르게 부르는 브랜드명 같은 경우의 안전망이 사라진다는 데
    합의). elevenst_raw는 _ElevenstFetchNode가 이 단계(propose_parallel)보다
    먼저 순차로 채워두므로 여기서 신뢰성 있게 읽을 수 있다. 그라운딩이
    이미 성공했으면 deepseek_raw를 빈 배열로 채운 채 건너뛰고(다른 propose
    슬롯이 실패했을 때와 동일한 "[]" 관례), 실패했을 때만(=진짜 필요할
    때만) 실제로 호출해 의미적 매칭으로 한 번 더 시도한다."""
    if _elevenst_grounded(callback_context.state):
        return _model_error_fallback_response("[]")
    return None


def _groq_model(model_name: str) -> LiteLlm:
    """"groq" 슬롯(config.py 참고, 2026-08-21부터 실제로는 HCX를 호출한다)은
    OpenAI 호환 엔드포인트라 "openai/" 프리픽스 뒤에 api_base/api_key로 그
    엔드포인트를 직접 지정한다 - litellm이 groq/hcx 프로바이더를 자체 지원하는지에
    기대지 않고 "그냥 OpenAI 호환 엔드포인트"로 취급하는 쪽이 확실하다(agents/gpt.py의
    DashScope 처리와 동일한 접근). num_retries=0 - 사용자 요청(2026-08-15: "너무
    느려 더 빠르게"): 실패해도 각 단계마다 이미
    폴백(on_model_error_callback 등)이 있어 litellm 내부 재시도로 얻는 이득보다
    지연 비용이 크다."""
    return LiteLlm(
        model=f"openai/{model_name}",
        api_base=settings.groq_api_base,
        api_key=settings.groq_api_key,
        num_retries=0,
    )


def _build_refine_agent() -> LlmAgent:
    def instruction(ctx: ReadonlyContext) -> str:
        return build_refine_query_prompt(ctx.state.get("original_query", ""))

    return LlmAgent(
        name="refine",
        # refine은 2026-08-16부터 Gemini가 아니라 Groq다(사용자 요청: "deepseek
        # Qwen 빼고 싹 다 무료 모델로 바꾸려고 해"). ADK가 output_schema를
        # response_format=json_schema로 요청하는데 propose 쪽 groq_model
        # (groq/compound-mini)은 이를 지원하지 않아, 구조화 출력이 되는 전용
        # 모델(groq_refine_model)을 따로 쓴다(config.py 주석 참고).
        model=_groq_model(settings.groq_refine_model),
        instruction=instruction,
        output_schema=RefinedQuery,
        output_key="refined_query",
        before_model_callback=_skip_refine_unless_conversational,
        on_model_error_callback=_on_refine_model_error,
    )


def _build_propose_agent(name: str, model, before_model_callback=None) -> LlmAgent:
    def instruction(ctx: ReadonlyContext) -> str:
        query = _refined_query_text(ctx.state)
        results = _search_results_from_state(ctx.state)
        return build_prompt(query, results)

    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        output_key=f"{name}_raw",
        on_model_error_callback=_on_propose_model_error,
        before_model_callback=before_model_callback,
    )


def _skip_challenge_if_all_structured(callback_context, llm_request) -> LlmResponse | None:
    """challenge(DeepSeek) 왕복을 건너뛸 수 있으면 건너뛴다(2026-08-20, 비용
    점검 - "LLM이 불필요하게 쓰이고 있는곳") - _apply_challenge()를 보면 후보의
    proposed_by에 구조화 소스(STRUCTURED_SOURCE_AGENTS - danawa/elevenst)가
    하나라도 있으면 challenge의 verdict를 통째로 무시하고 verified=True로
    강제한다(이미 실측 확인된 가격을 텍스트 기반 LLM에게 다시 판단하게 하는
    건 불필요하다는 게 그 검증 로직 자체의 전제). 11번가가 이제 유일한 검색
    소스라, elevenst 그라운딩이 성공하면 propose의 유일한 LLM(deepseek)조차
    조건부로 건너뛰므로(_skip_propose_if_elevenst_grounded) 병합된 후보가
    elevenst 하나뿐인 경우가 흔하다 - 이때는 DeepSeek(challenge) 호출 자체가
    순수 낭비다. 후보가 하나도 없어도(검증할 대상 자체가 없음) 같은 이유로
    건너뛴다. verdicts를 빈 배열로 채운 채 넘기면 _apply_challenge가 challenge를
    실제로 돌렸을 때와 동일하게 동작한다(구조화 소스 후보는 애초에 verdicts를
    안 보고, 나머지는 없으므로 영향 없음). 후보 중 하나라도 구조화 소스가
    아니면(=진짜 검증이 필요한 후보가 섞여 있으면) None을 반환해 실제 challenge를
    그대로 태운다."""
    candidates: list[dict] = callback_context.state.get("candidates") or []
    if not candidates or all(
        STRUCTURED_SOURCE_AGENTS & set(c.get("proposed_by") or []) for c in candidates
    ):
        return _model_error_fallback_response("[]")
    return None


def _build_challenge_agent() -> LlmAgent:
    def instruction(ctx: ReadonlyContext) -> str:
        query = _refined_query_text(ctx.state)
        candidates = ctx.state.get("candidates") or []
        results = _search_results_from_state(ctx.state)
        candidate_pages = ctx.state.get("candidate_pages") or {}
        coupang_results = [SearchResult(**r) for r in ctx.state.get("coupang_results") or []]
        naver_results = [SearchResult(**r) for r in ctx.state.get("naver_results") or []]
        return build_challenge_prompt(query, candidates, results, candidate_pages, coupang_results, naver_results)

    return LlmAgent(
        name="challenge",
        model=LiteLlm(model=f"deepseek/{settings.deepseek_model}", num_retries=0),
        instruction=instruction,
        output_key="raw_challenge",
        before_model_callback=_skip_challenge_if_all_structured,
    )


def _judge_eligible_proposals(proposals: list[Proposal]) -> list[Proposal]:
    """DeepSeek 검증에서 명확히 우려(verified=False)로 표시된 후보는 judge에게
    아예 보여주지 않는다 — 예전에는 참고 정보로만 주고 LLM이 그래도 우려 후보를
    고를 수 있는 구조였다. 단, 전부 우려로 표시된 경우(검증 자체가 과도하게
    엄격했을 가능성)에는 예외적으로 전체 목록을 그대로 넘긴다 — 하나도 못
    고르는 것보다는 최선의 후보라도 고르는 게 낫다."""
    eligible = [p for p in proposals if p.verified is not False]
    return eligible or proposals


def _skip_judge_if_single_candidate(callback_context, llm_request) -> LlmResponse | None:
    """검증을 통과한 제안이 정확히 1개면 그 사이에서 "고를" 게 없다 - judge(Claude,
    약 5초) 호출을 건너뛰고 그 하나를 그대로 채택한다(사용자 요청, 2026-08-15:
    "너무 느려 더 빠르게" - 에이전트당 최종 후보 1개 + dedup 병합 + verified=False
    탈락을 거치고 나면, 특히 지금처럼 일부 에이전트 토큰이 소진된 상황에서는
    실제로 이 경우가 자주 나온다). _build_decision()이 judge의 raw product_name/
    price/retailer보다 url로 찾은 실제 proposal(matched)의 그라운딩된 값을
    우선하므로, 여기서는 url만 정확히 일치시키면 나머지 필드는 자동으로 검증된
    값으로 채워진다 - reasoning만 이 함수가 직접 채운다.

    후보가 0개거나 2개 이상이면 원래대로 judge를 그대로 태운다 - 0개는 "고를
    필요 없음"이 아니라 "아무것도 못 찾음"이라 judge의 완화된 자유 추론이 여전히
    의미 있고, 2개 이상은 진짜 비교/심사가 필요하다."""
    proposals = [Proposal(**p) for p in (callback_context.state.get("proposals") or [])]
    eligible = _judge_eligible_proposals(proposals)
    if len(eligible) != 1:
        return None
    only = eligible[0]
    if not (only.product_name and only.price and only.retailer and only.url):
        return None
    verdict = judge_module.JudgeVerdict(
        product_name=only.product_name,
        price=only.price,
        retailer=only.retailer,
        url=only.url,
        reasoning=f"다른 유효한 제안이 없어 {only.agent}의 후보를 그대로 채택했습니다.",
    )
    return _model_error_fallback_response(verdict.model_dump_json())


def _build_judge_agent() -> LlmAgent:
    def instruction(ctx: ReadonlyContext) -> str:
        query = _refined_query_text(ctx.state)
        proposals = [Proposal(**p) for p in (ctx.state.get("proposals") or [])]
        eligible = _judge_eligible_proposals(proposals)[:_MAX_JUDGE_INPUT_CANDIDATES]
        return judge_module.build_judge_prompt(query, eligible)

    return LlmAgent(
        name="judge",
        # judge는 2026-08-16부터 Claude가 아니라 "groq" 슬롯이다(사용자 요청:
        # "deepseek Qwen 빼고 싹 다 무료 모델로 바꾸려고 해" - Anthropic엔 상시
        # 무료 API 티어가 없다). 그 슬롯이 실제로 부르는 모델은 이후 여러 번
        # 바뀌었다(Groq gpt-oss-120b → 2026-08-21 HCX-005 - config.py의
        # groq_judge_model 주석 참고) - output_schema(구조화 출력)를 지원하는
        # 모델이어야 한다는 제약은 그대로 유지된다.
        model=_groq_model(settings.groq_judge_model),
        instruction=instruction,
        output_schema=judge_module.JudgeVerdict,
        output_key="raw_decision",
        before_model_callback=_skip_judge_if_single_candidate,
    )


def _build_pipeline() -> SequentialAgent:
    deepseek_raw = "deepseek"

    propose_parallel = ParallelAgent(
        name="propose",
        sub_agents=[
            # (2026-08-20, "3개 LLM까지 필요없다") gpt(Qwen)/groq 슬롯은 여기서
            # 뺐다 - 검색이 이제 11번가 하나뿐이라 세 LLM이 전부 _ElevenstFetchNode
            # (아래, 이제 이 병렬 블록보다 먼저 순차 실행됨)와 완전히 같은 데이터를
            # 보고 있었고, 그 구조화 데이터를 다시 텍스트로 추측해 읽는 것보다
            # elevenst의 정확한 가격 필드를 그대로 쓰는 게 낫다. _build_propose_agent
            # 자체는 그대로 남아있어 필요하면 언제든 다시 추가할 수 있다(gpt_raw
            # 슬롯 관련 배경 코멘트는 git 히스토리 참고). deepseek 하나만 남기고,
            # elevenst 그라운딩이 실패했을 때만(오타/비속어/다르게 부르는 브랜드명
            # 등 rapidfuzz 매칭이 못 잡는 경우의 의미적 안전망) 조건부로 호출한다.
            _build_propose_agent(
                deepseek_raw,
                LiteLlm(model=f"deepseek/{settings.deepseek_model}", num_retries=0),
                before_model_callback=_skip_propose_if_elevenst_grounded,
            ),
            # 쿠팡 교차 확인(2026-08-16, "그라운딩 성능을 높여줘") - 후보를 만들지
            # 않고 challenge 단계의 참고 신호만 채운다. 같은 ParallelAgent 소속이라
            # 지연시간이 추가되지 않는다. elevenst 그라운딩 성공 시 자체적으로
            # 건너뛴다(_CoupangCheckNode 참고 - challenge가 그 결과를 안 볼 것이므로).
            _CoupangCheckNode(name="coupang_check"),
            # 네이버쇼핑 교차 확인(2026-08-16, "다나와 단일 실측 소스에 대한
            # 의존도를 낮추도록") - 쿠팡과 같은 소프트 신호 패턴의 두 번째 소스.
            _NaverCheckNode(name="naver_check"),
        ],
    )

    return SequentialAgent(
        name="single_debate_pipeline",
        sub_agents=[
            # refine(질의 정제) 단계는 2026-08-20("쿼리 재질의 없애고")에 한 번
            # 뺐었는데, 같은 날 뒤이어 "'안녕 나 컵을 사고싶어' 이런식으로
            # 쿼리를 입력하면 지금 LLM이 못 알아듣거든" 리포트로 회귀가 드러나
            # 다시 넣었다 - 대화체/인사말이 섞인 질의는 정제 없이 그대로 11번가
            # keyword 검색에 들어가면 검색·그라운딩이 둘 다 실패한다. 예전과
            # 달리 매 질의마다 도는 게 아니라 looks_conversational_query()에
            # 걸리는 질의에만 조건부로 돈다(_skip_refine_unless_conversational) -
            # "음료수"처럼 이미 짧고 깨끗한 검색어는 여전히 이 단계를 건너뛴다.
            _build_refine_agent(),
            _SearchNode(name="search"),
            # 11번가 공식 API 구조화 가격 - 2026-08-20("11번가 api를 구해서
            # 다나와를 폐기하고 11번가 쪽으로 방향을 틀려고") - 다나와 스크래핑의
            # 불안정함(AWS IP 차단, search.danawa.com Crawl-delay) 대신 공식
            # 오픈API로 구조화된 후보를 만든다. _DanawaFetchNode 자체는 코드로
            # 남겨뒀다(참고/롤백 대비). (2026-08-20 재구성) propose_parallel과
            # 같은 자리에 있었으나, propose/coupang_check/naver_check가 이 결과를
            # 보고 자신을 건너뛸지 정해야 해서 그보다 먼저 도는 별도 순차 단계로
            # 옮겼다(_ElevenstFetchNode 문서 참고).
            _ElevenstFetchNode(name="elevenst"),
            propose_parallel,
            _FilterMergeNode(name="filter_merge"),
            _ExtractPagesNode(name="extract_pages"),
            _build_challenge_agent(),
            _ApplyChallengeNode(name="apply_challenge"),
            _build_judge_agent(),
        ],
    )


_runner: InMemoryRunner | None = None


def _get_runner() -> InMemoryRunner:
    global _runner
    if _runner is None:
        _runner = InMemoryRunner(agent=_build_pipeline(), app_name=_APP_NAME)
    return _runner


_STAGE_AFTER = {
    "refine": "searching",
    "search": "proposing",
    "filter_merge": "challenging",
    "apply_challenge": "judging",
}

_decide_result_adapter = TypeAdapter(DecideResponse | ClarifyResponse)


def _build_decision(state: dict, proposals: list[Proposal]) -> Decision | None:
    """judge(LLM)는 raw_decision에 자기 나름의 product_name/price/retailer/url을
    자유 텍스트로 다시 쓸 수 있는데, 후보 목록에 없는 값을 지어낼 위험이 있다
    (실제로 url이 빈 후보를 골랐을 때 judge가 그럴듯한 URL을 스스로 채워 넣는
    사례를 확인했다). raw.url로 실제 후보(matched)를 찾은 뒤에는, 검증을 거친
    matched의 값(그라운딩된 데이터)을 우선하고 raw는 matched에 값이 없을 때만
    보충으로 쓴다 — reasoning만은 judge가 직접 작성하는 문장이라 그대로 쓴다."""
    raw = state.get("raw_decision")
    if not raw or not proposals:
        return None

    url = raw.get("url")
    matched = next((p for p in proposals if p.url and p.url == url), None) or proposals[0]

    return Decision(
        product_name=matched.product_name or raw.get("product_name") or "",
        price=matched.price or raw.get("price") or "",
        retailer=matched.retailer or raw.get("retailer") or "",
        url=matched.url or raw.get("url") or "",
        reasoning=raw.get("reasoning") or "",
        chosen_agent=matched.agent,
        verified=matched.verified,
    )


async def _build_style_guide(query: str, proposals: list[Proposal]) -> StyleGuide | None:
    """취향 주도 카테고리(STYLE_GUIDE_CATEGORIES)에서만, judge가 하나 고른
    최종 추천(decision)과는 별개로 challenge를 통과한 proposals 전체를
    스타일별로 그룹핑해 덧붙인다(2026-08-19 사용자 요청 - GPT 쇼핑처럼
    여러 검증된 후보를 스타일별로 보여주되, 최종 추천은 지금처럼 하나로
    유지). judge_module.generate_style_guide는 LLM 호출+파싱만 하고, 여기서
    _build_decision과 같은 그라운딩 검증을 한다 - LLM이 목록에 없는 url을
    지어내면(judge에서 이미 한 번 확인된 위험) 그 그룹은 조용히 버린다.
    그룹이 2개 미만으로 남으면(대부분 무관하다고 판단됐거나 실패) 아예
    None - 스타일 가이드는 있으면 좋은 부가 정보일 뿐, 없어도 기존 단일
    추천 응답과 완전히 동일하게 동작해야 한다."""
    eligible = _judge_eligible_proposals(proposals)[:_MAX_JUDGE_INPUT_CANDIDATES]
    if len(eligible) < 2:
        return None
    try:
        data = await judge_module.generate_style_guide(query, eligible)
    except Exception:
        logger.exception("스타일 가이드 생성 실패: %r", query)
        return None

    by_url = {p.url: p for p in eligible if p.url}
    groups: list[StyleGuideGroup] = []
    for g in data.get("groups") or []:
        url = g.get("url")
        if url not in by_url:
            continue
        label = (g.get("label") or "").strip()
        description = (g.get("description") or "").strip()
        if not label or not description:
            continue
        groups.append(StyleGuideGroup(label=label, description=description, url=url))
    if len(groups) < 2:
        return None
    return StyleGuide(intro=data.get("intro") or "", groups=groups, closing_pick=data.get("closing_pick"))


async def _verify_relaxed_verdict(
    query: str, verdict: judge_module.JudgeVerdict, search_results: list[SearchResult]
) -> bool | None:
    """relaxed fallback이 고른 verdict를 정상 propose 후보와 동일한 DeepSeek
    challenge 기준으로 그라운딩 재검증한다(2026-08-16 하드닝 - "구매링크를
    안띄워주는거야" 버그의 근본 원인이 바로 이 경로였다: challenge/CMPNYC_MAP
    검증을 전부 우회한 채 Qwen 단독 판단을 그대로 최종 응답으로 내보냈다).
    쿠팡·네이버쇼핑 참고 신호도 정상 challenge와 동일하게 함께 준다. 검증
    자체가 실패(API 오류·파싱 실패)하면 None(미검증)을 반환한다 - 검증 인프라
    장애 때문에 이미 찾은 유일한 후보를 버리지 않는다."""
    candidate = {
        "product_name": verdict.product_name,
        "price_krw": verdict.price,
        "retailer": verdict.retailer,
        "url": verdict.url,
        "reasoning": verdict.reasoning,
    }
    coupang_task = search_module.search_coupang(query)
    naver_task = search_module.search_naver(query)
    coupang_results, naver_results = await asyncio.gather(coupang_task, naver_task)
    verdicts = await deepseek_module.challenge_candidates(
        query, [candidate], search_results, None, coupang_results, naver_results
    )
    if not verdicts:
        return None
    match = next((v for v in verdicts if v.url == verdict.url), verdicts[0])
    return match.verified


async def _pick_and_verify_relaxed(
    query: str, search_results: list[SearchResult]
) -> tuple[judge_module.JudgeVerdict, bool | None] | None:
    """한 라운드(주어진 검색 결과 기준)의 relaxed pick + challenge 재검증을
    묶은 헬퍼. verified=False로 명백히 우려 표시되면 그 후보는 폐기하고
    None을 돌려줘(호출부가 다음 라운드로 넘어가거나 완전히 포기하게) - "낮은
    확신"과 "명백히 틀림"은 다르게 취급한다(challenge의 기존 원칙과 동일:
    애매하면 false 남발 금지, 명백할 때만 false)."""
    verdict = await gpt_module.pick_most_relevant(query, search_results)
    if verdict is None:
        return None
    verified = await _verify_relaxed_verdict(query, verdict, search_results)
    if verified is False:
        logger.info("relaxed fallback 후보가 challenge 검증에서 탈락 — 폐기: %r", verdict.url)
        return None
    return verdict, verified


async def _relaxed_fallback_decision(query: str, search_results: list[SearchResult]) -> Decision | None:
    """적절한 후보를 하나도 못 찾았을 때의 폴백(2026-08-15, "적절한 상품 후보를
    찾지 못하면 다시 fallback해서 feedback 구조로 돌아가서 가장 관련성 높은
    상품을 추천해주는 시스템으로 가고 싶어" - 완전 재검색 루프(끝없이 검색어를
    넓혀가는 것)와 단순 1회 재시도의 중간): 최대 두 라운드만 시도하고 그래도
    없으면 정직하게 포기한다(NO_CANDIDATE_ERROR).

    1라운드 - 이미 가져온 검색 결과 그대로, gpt.pick_most_relevant로 완벽히
    일치하지 않아도 가장 관련성 높은 것을 고르게 한 뒤, DeepSeek challenge로
    다시 검증한다(2026-08-16 하드닝). 검색 자체는 다시 안 하므로 비용이
    거의 없다.

    2라운드(1라운드가 후보를 아예 못 찾았거나 challenge에서 명백히 탈락했을
    때만) - 질의를 앞 2단어로 넓혀 한 번만 재검색한 뒤 같은 기준으로 다시
    시도한다. 넓힌 질의가 원래 질의와 같으면(이미 짧은 질의라 넓힐 게 없으면)
    똑같은 검색을 반복하는 낭비이므로 건너뛴다."""
    picked = await _pick_and_verify_relaxed(query, search_results)
    used_broadened = False
    if picked is None:
        tokens = query.split()
        broadened_query = " ".join(tokens[:2]) if len(tokens) > 2 else query
        if broadened_query != query:
            try:
                broadened_results = await search_module.search(broadened_query)
            except Exception:
                broadened_results = []
            picked = await _pick_and_verify_relaxed(query, broadened_results)
            used_broadened = True
    if picked is None:
        return None
    verdict, verified = picked
    if not verdict.product_name:
        return None

    reasoning = verdict.reasoning
    if used_broadened:
        reasoning = f"'{query}'로는 적절한 상품을 찾지 못해 검색 범위를 넓혀 찾았습니다. {reasoning}"
    if verified is not True:
        reasoning = f"{reasoning} (참고: 교차 검증을 통과하지 못한 낮은 확신의 답변입니다.)"
    return Decision(
        product_name=verdict.product_name,
        price=verdict.price,
        retailer=verdict.retailer,
        url=verdict.url,
        reasoning=reasoning,
        chosen_agent="gpt",
        verified=verified,
    )


async def _is_relevant_to_query(query: str, product_name: str) -> bool:
    """이 상품이 이 검색어가 찾는 상품 "종류"와 실제로 맞는지 LLM 한 번으로
    확인한다 - _comparison_page_listing_fallback 전용. 텍스트 유사도
    (rapidfuzz token_set_ratio)로는 이 판정이 안 됐다(실측 2026-08-19:
    "10만원대 이어폰 추천해줘"에 상품명 자체가 카테고리 단어를 안 담은
    무관한 상품(쌍안경 "니쿠라 10-30x25", 노트북 "삼성전자 갤럭시북2 프로")이
    가격/순서 기준 1위로 뽑혔는데, 유사도 점수로는 진짜 이어폰들과 구분이
    안 됐다 - 제품명이 브랜드+모델명뿐이라 카테고리 단어 자체가 아예 없는
    경우가 흔해서 문자열 매칭 자체가 이 문제엔 안 맞았다). 아주 드물게
    (다른 모든 경로가 실패했을 때만) 타는 최후 폴백이라 LLM 호출 하나를
    추가로 쓸 여유가 있다.

    groq_refine_model(gpt-oss-20b)이 아니라 groq_judge_model(gpt-oss-120b)을
    쓴다 - 실측(2026-08-19): 같은 프롬프트로 "호카 클리프톤 10"(러닝화)이
    "10만원대 이어폰" 검색과 relevant=true라고 확신에 차 틀린 답을 낸 걸
    발견했다. 20b는 이 판정 자체를 못 하는 것으로 보이고, 이미 judge/propose
    단계에서 신뢰도가 검증된 120b로 바꾸니 즉시 올바르게(false) 판정했다."""
    prompt = (
        "다음 상품이 이 검색어가 찾는 상품 종류와 실제로 맞는지만 판단하세요. "
        "가격대·브랜드 선호·재고는 무시하고, 상품 카테고리/종류 자체가 맞는지만 보세요. "
        '반드시 {"relevant": true 또는 false} 형식의 JSON으로만 답하세요.\n\n'
        f"검색어: {query}\n상품명: {product_name}"
    )
    try:
        client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_api_base, max_retries=0)
        response = await client.chat.completions.create(
            model=settings.groq_judge_model,
            messages=[{"role": "user", "content": prompt}],
        )
        data = parse_json_object(response.choices[0].message.content or "")
        return bool(data.get("relevant"))
    except Exception:
        logger.exception("비교 페이지 폴백 관련성 판정 실패: %r / %r", query, product_name)
        return False


async def _comparison_page_listing_fallback(
    query: str, danawa_tables: list[tuple[PriceTable, dict]]
) -> Decision | None:
    """직접 구매 링크를 만들 수 있는 후보가 하나도 없을 때(propose 3곳 +
    _relaxed_fallback_decision까지 전부 실패)의 최후 폴백(사용자 요청,
    2026-08-19: "그래도 추천해줘라고 했을때 답변을 잘해주는거잖아") - 다나와가
    실측한 가격표 자체는 있는데(CMPNYC_MAP에 없거나 url_rule=None인 판매처만
    걸려 A등급 구매링크를 못 만드는 경우, 예: 쿠팡 제휴 코드가 막힌 상태)
    "아무것도 못 찾았다"고 끝내는 대신, 다나와 가격비교 페이지 자체를(실제
    존재하는 페이지, 실제 최저가) 정직하게 보여준다.

    danawa_tables는 표본 자체가 작고 노이즈가 많다(실측: url 최대 5개 중
    fetch_danawa_offers가 실제로 성공하는 건 보통 1개뿐이라 "순서"나
    "가격" 같은 값싼 휴리스틱으로는 관련성을 걸러낼 수 없었다 - 최저가
    기준도, 첫 번째 기준도 둘 다 무관한 상품을 골랐다). 그래서 표마다
    _is_relevant_to_query로 실제 확인해, 통과하는 첫 번째 표만 쓴다 -
    전부 무관하면(또는 모든 표가 offers/pcode가 없으면) 정직하게 None을
    반환해 호출부가 NO_CANDIDATE_ERROR로 이어지게 한다 - 무관한 상품을
    추천하느니 못 찾았다고 하는 게 낫다.

    최종 URL/판매처는 main.py::_resolve_danawa_urls(danawa.resolve_lowest_price)가
    한 번 더 다나와 자체 AJAX로 실제 최저가 판매처를 조회해 바꿔치기할 수도,
    실패하면 이 비교 페이지 URL 그대로 둘 수도 있다 - 그래서 reasoning은 둘
    중 어느 쪽이 되어도 거짓이 되지 않게 "직접 검증된 후보가 없어 다나와
    실측 데이터를 대신 안내한다"는 사실만 말하고, "링크를 못 만들었다"처럼
    나중에 틀릴 수 있는 단정은 하지 않는다.

    호출부는 이 결과를 _finalize_with_danawa에 넘기지 않는다 -
    exclude_price_comparison_site_as_final_pick이 정확히 이런 비교 페이지
    URL을 최종 결정에서 걷어내는 역할이라(정상 경로에서는 맞는 동작) 여기서
    의도적으로 만드는 비교 페이지 폴백까지 걷어내 버린다."""
    candidates = [(pt, dr) for pt, dr in danawa_tables if pt.offers and pt.source_pcode]
    for price_table, _ in candidates:
        if not await _is_relevant_to_query(query, price_table.product_name or ""):
            continue
        cheapest = price_table.offers[0]
        return Decision(
            product_name=price_table.product_name or "상품",
            price=_format_price_krw(cheapest.price_krw),
            retailer="다나와 가격비교",
            url=f"https://prod.danawa.com/info/?pcode={price_table.source_pcode}",
            reasoning=(
                "AI 제안 중 직접 검증된 구매 후보가 없어, 다나와가 실측한 가격비교 데이터를 "
                f"대신 안내합니다 - {cheapest.seller} 등 {len(price_table.offers)}개 판매처의 "
                "실제 가격이 확인됩니다."
            ),
            chosen_agent="danawa",
            price_source="danawa_offer",
            verified=True,
        )
    return None


def _danawa_tables_from_state(state: dict) -> list[tuple[PriceTable, dict]]:
    return [(PriceTable(**t), raw) for t, raw in (state.get("danawa_tables") or [])]


async def _finalize_with_danawa(
    decision: Decision,
    proposals: list[Proposal],
    danawa_tables: list[tuple[PriceTable, dict]],
) -> tuple[Decision, PriceTable | None]:
    """judge(또는 relaxed fallback)가 고른 decision을 다나와 실측 데이터로
    후처리한다 - PRESERVED FROM seungmin/lsm의 run_single_debate_price_table_variant
    PART 4-2/4-3과 동일한 순서(debate.py 260-273줄)를 ADK 파이프라인으로 포팅.
    danawa_tables가 비어있으면(다나와 조회 실패/결과 없음) 세 호출 모두
    안전하게 아무것도 안 하고 decision을 그대로 돌려준다.

    1) judge가 실제로 danawa 후보를 골랐으면 price_source를 danawa_offer로 표시.
    2) pick_primary(offer가 가장 많은 페이지)를 응답의 price_table로 채운다 -
       지금까지 ADK 파이프라인은 이 필드를 항상 null로 냈다.
    3) enrich_decision - judge가 고른 상품명이 다나와 페이지와 이름이 맞으면
       실측 가격/판매처/URL로 덮어쓴다(이름이 안 맞으면 손대지 않음).
    4) exclude_price_comparison_site_as_final_pick - 최종 URL이 다나와
       가격비교 페이지 자체면(judge가 실수로 그렇게 골랐을 때) 실제 구매
       URL로 치환하거나 다른 제안으로 넘긴다."""
    if decision.chosen_agent == "danawa":
        decision.price_source = "danawa_offer"
    elif decision.chosen_agent == "elevenst":
        decision.price_source = "elevenst_offer"

    primary = price_table_module.pick_primary(danawa_tables)
    price_table: PriceTable | None = None
    if primary is not None:
        table, raw_result = primary
        price_table = table
        decision = await price_table_module.enrich_decision(decision, raw_result)

    decision = await price_table_module.exclude_price_comparison_site_as_final_pick(
        decision, proposals, danawa_tables
    )
    return decision, price_table


async def run_stream(
    query: str, skip_clarify: bool = False, persona: dict[str, str] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """run()과 같은 결과를 만들지만, 단계마다 NDJSON 이벤트를 흘려보낸다 —
    debate.py::run_single_debate_stream이 그대로 재노출.

    persona(2026-08-20, check_clarify_facets 사전 호출 제거의 부작용 방지) -
    /decide/clarify 사전 호출이 없어지면서, 페르소나 기반 facet 순서 반영이
    유지되려면 이 파이프라인 내부의 clarify 안전망(_extract_clarify_options
    호출부, 아래 두 곳)에도 페르소나가 도달해야 한다.

    skip_clarify(2026-08 통합 병합) - 프론트의 SearchContext.runTurn이 이미
    브랜드/facet/고정축 선택으로 한 라운드를 좁혀온 후속 턴이면 True로 넘어온다
    (main.py의 DecideRequest.skip_intent_check). 이게 없으면 검색 직후
    debate._is_ambiguous_facets()가 이번 라운드에도 남아있는 다른 facet(예:
    브랜드는 답했는데 용량이 여러 개)을 또 clarify로 멈춰세워, 사용자가 이미
    몇 라운드나 답했는데도 계속 새 선택 화면이 뜨는 재질문 버그가 생긴다 -
    True면 이 조기 종료를 건너뛰고 바로 propose/challenge/judge까지 진행한다
    (완전히 후보가 0개면 아래의 안전망 clarify는 skip_clarify와 무관하게
    그대로 동작한다)."""
    from .debate import _extract_clarify_options, _is_ambiguous_facets  # 지연 임포트 — 순환 참조 방지

    runner = _get_runner()
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=_APP_NAME, user_id="anonymous", session_id=session_id, state={"original_query": query}
    )

    # refine이 다시 첫 단계다(2026-08-20, 조건부 재도입) - "refining" 상태를
    # 먼저 보여준다. 대부분의 질의는 이 단계가 즉시 스킵되지만(_skip_refine_
    # unless_conversational), LlmAgent 노드 자체는 여전히 이벤트를 내므로
    # _STAGE_AFTER["refine"]="searching"이 정상적으로 다음 상태로 넘겨준다.
    yield {"type": "status", "stage": "refining"}

    pipeline_failed = False
    gen = runner.run_async(
        user_id="anonymous",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    )
    try:
        async for event in gen:
            # search도 우리가 직접 만든 커스텀 노드라 이벤트 모양을 확실히 안다.
            # 검색 직후 브랜드/용량/개수가 애매하면(Human-in-the-loop) 제안 단계로
            # 넘어가기 전에 여기서 파이프라인을 실제로 멈춘다 — 제안/검증/심사는
            # 아예 실행되지 않는다(만들어놓고 버리는 게 아니라 애초에 안 돈다).
            if event.author == "search" and event.actions and event.actions.state_delta:
                search_results = [
                    SearchResult(**r) for r in event.actions.state_delta.get("search_results") or []
                ]
                # skip_clarify=True면 아래 조건이 항상 거짓이라 clarify 결과를
                # 쓰지 않는다 - 그런데도 호출 자체는 무조건 나가고 있었다.
                # _extract_clarify_options는 브랜드별 최대 15개 병렬 DeepSeek
                # 호출까지 갈 수 있는 무거운 함수라(debate.py 참고), 후속 라운드
                # (skip_clarify=True)마다 결과를 100% 버리면서도 이 비용을 그대로
                # 치르고 있었다 - 토큰 절약(2026-08-19)으로 아예 호출을 건너뛴다.
                clarify = (
                    None
                    if skip_clarify
                    else await _extract_clarify_options(query, search_results, persona)
                )
                if (
                    clarify is not None
                    and _is_ambiguous_facets(query, clarify.options.facets)
                ):
                    await gen.aclose()
                    yield {"type": "final", "result": clarify.model_dump()}
                    return

            # apply_challenge는 우리가 직접 만든 커스텀 노드라 이 이벤트의
            # state_delta 형태를 확실히 안다(LlmAgent의 output_key 이벤트 형태는
            # 검증되지 않아 의존하지 않음 — 최종 상태는 아래에서 get_session()으로
            # 한 번에 확정해서 읽는다). 후보를 먼저 보여준 뒤 judging 단계로 넘어가야
            # 하므로 proposal 이벤트를 status보다 먼저 yield한다.
            if event.author == "apply_challenge" and event.actions and event.actions.state_delta:
                for proposal in event.actions.state_delta.get("proposals") or []:
                    yield {"type": "proposal", "proposal": proposal}

            next_stage = _STAGE_AFTER.get(event.author)
            if next_stage:
                yield {"type": "status", "stage": next_stage}
    except Exception:
        logger.exception("ADK 파이프라인 실행 실패: %r", query)
        pipeline_failed = True

    final_session = await runner.session_service.get_session(
        app_name=_APP_NAME, user_id="anonymous", session_id=session_id
    )
    final_state: dict = dict(final_session.state) if final_session else {}
    if pipeline_failed:
        final_state.setdefault("proposals", [])

    danawa_tables = _danawa_tables_from_state(final_state)
    proposals = [Proposal(**p) for p in (final_state.get("proposals") or [])]

    if not proposals:
        search_results = _search_results_from_state(final_state)
        clarify = await _extract_clarify_options(query, search_results, persona)
        if clarify is not None:
            yield {"type": "final", "result": clarify.model_dump()}
            return
        fallback_decision = await _relaxed_fallback_decision(query, search_results)
        if fallback_decision is not None:
            fallback_decision, price_table = await _finalize_with_danawa(fallback_decision, [], danawa_tables)
            result = DecideResponse(
                query=query, proposals=[], decision=fallback_decision, price_table=price_table
            )
            yield {"type": "final", "result": result.model_dump()}
            return
        listing_decision = await _comparison_page_listing_fallback(query, danawa_tables)
        if listing_decision is not None:
            primary = price_table_module.pick_primary(danawa_tables)
            result = DecideResponse(
                query=query,
                proposals=[],
                decision=listing_decision,
                price_table=primary[0] if primary else None,
            )
            yield {"type": "final", "result": result.model_dump()}
            return
        raise RuntimeError(NO_CANDIDATE_ERROR)

    decision = _build_decision(final_state, proposals)
    if decision is None:
        fallback_decision = await _relaxed_fallback_decision(query, _search_results_from_state(final_state))
        if fallback_decision is not None:
            fallback_decision, price_table = await _finalize_with_danawa(
                fallback_decision, proposals, danawa_tables
            )
            result = DecideResponse(
                query=query, proposals=proposals, decision=fallback_decision, price_table=price_table
            )
            yield {"type": "final", "result": result.model_dump()}
            return
        listing_decision = await _comparison_page_listing_fallback(query, danawa_tables)
        if listing_decision is not None:
            primary = price_table_module.pick_primary(danawa_tables)
            result = DecideResponse(
                query=query,
                proposals=proposals,
                decision=listing_decision,
                price_table=primary[0] if primary else None,
            )
            yield {"type": "final", "result": result.model_dump()}
            return
        raise RuntimeError(NO_CANDIDATE_ERROR)

    decision, price_table = await _finalize_with_danawa(decision, proposals, danawa_tables)
    style_guide = (
        await _build_style_guide(query, proposals) if final_state.get("style_guide_mode") else None
    )
    result = DecideResponse(
        query=query,
        proposals=proposals,
        decision=decision,
        price_table=price_table,
        style_guide=style_guide,
    )
    yield {"type": "final", "result": result.model_dump()}


async def run(
    query: str, skip_clarify: bool = False, persona: dict[str, str] | None = None
) -> DecideResponse | ClarifyResponse:
    result_dict = None
    async for event in run_stream(query, skip_clarify=skip_clarify, persona=persona):
        if event["type"] == "final":
            result_dict = event["result"]
    if result_dict is None:
        raise RuntimeError(NO_CANDIDATE_ERROR)
    return _decide_result_adapter.validate_python(result_dict)
