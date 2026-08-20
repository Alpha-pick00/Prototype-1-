import asyncio
import logging
import re
from typing import Any, AsyncIterator

from fetchers import danawa, danawa_search, elevenst

from . import adk_pipeline
from . import decision_cache
from . import facet_cache
from . import price_table as price_table_module
from . import search as search_module
from .agents import deepseek, gpt, groq, judge
from .agents.base import NO_CANDIDATE_ERROR, is_generic_listing_url
from .config import settings
from .intent import is_bulk_query, is_non_product_chitchat, needs_clarification
from .schemas import (
    BrandOption,
    BrandPriceResponse,
    BulkDecideResponse,
    BulkDecision,
    BulkProposal,
    ClarifyFacet,
    ClarifyOptions,
    ClarifyResponse,
    Decision,
    DecideResponse,
    PriceRange,
    Proposal,
    SearchResult,
)

logger = logging.getLogger(__name__)


def _price_to_int(price: str) -> int | None:
    digits = re.sub(r"[^\d]", "", price or "")
    return int(digits) if digits else None


def _compute_price_range(options: list[BrandOption]) -> PriceRange | None:
    prices = [n for n in (_price_to_int(o.price) for o in options) if n is not None]
    if not prices:
        return None
    return PriceRange(min=f"{min(prices):,}원", max=f"{max(prices):,}원")


def _any_llm_key_configured() -> bool:
    return bool(settings.qwen_api_key or settings.groq_api_key or settings.deepseek_api_key)


async def run_debate(query: str) -> DecideResponse | BulkDecideResponse | ClarifyResponse:
    if is_non_product_chitchat(query):
        # 검색/LLM 호출을 아예 안 하고 즉시 실패한다 - is_non_product_chitchat 참고.
        raise RuntimeError(NO_CANDIDATE_ERROR)
    static_decision = decision_cache.lookup(query)
    if static_decision is not None:
        return DecideResponse(**static_decision)
    if is_bulk_query(query):
        return await run_bulk_debate(query)
    if not _any_llm_key_configured():
        # 임시(로컬 실험용) - LLM 키가 하나도 없으면 /decide 자체를 다나와
        # 전용 규칙 기반 경로로 자동 전환한다. .env에 키를 하나라도 채우고
        # 재시작하면 이 분기를 안 타고 원래 LLM 파이프라인으로 바로 돌아간다.
        # needs_clarification()보다 먼저 검사해야 한다 - run_clarify()는 GPT를
        # 호출하므로, 키가 아예 없을 때 그리로 먼저 새면 실패한다(2026-08-12,
        # needs_clarification()이 짧은 검색어까지 잡도록 넓히면서 드러난 순서
        # 버그 - 이 분기가 없었을 땐 우연히 순서가 안 겹쳤을 뿐이었다).
        return await run_danawa_only_debate(query)
    if needs_clarification(query):
        return await run_clarify(query)
    return await run_single_debate(query)


async def run_debate_stream(query: str) -> AsyncIterator[dict[str, Any]]:
    """단일 상품 검색만 단계별 이벤트로 스트리밍한다. bulk/clarify는 흐름 자체가
    다르고(브랜드 목록 선택, 가격대별 정리) 단계를 쪼갤 만한 지점이 마땅치 않아,
    최종 결과 하나만 "final" 이벤트로 보낸다 — 프론트는 이벤트 타입 하나만 보고
    두 경우 다 처리하면 된다."""
    if is_non_product_chitchat(query):
        yield {"type": "error", "message": NO_CANDIDATE_ERROR}
        return
    static_decision = decision_cache.lookup(query)
    if static_decision is not None:
        # 정적 최종결과 캐시(2026-08-16, 속도 개선) - "여기서 상세검색까지
        # 누르면 바로 정규식으로 찾을수있게 0.1초만에 해줘": AI 상세검색에서
        # facet을 다 고른 뒤 넘어가는 최종 검색(정제+검색+제안 3개+검증+심사
        # 전체, 실측 ~9~15초)을 건너뛰고 캡처해둔 실제 결과를 즉시 낸다.
        yield {"type": "final", "result": static_decision}
        return
    if is_bulk_query(query):
        yield {"type": "final", "result": (await run_bulk_debate(query)).model_dump()}
        return
    if not _any_llm_key_configured():
        # run_debate()의 "LLM 키가 하나도 없으면 다나와 전용 경로로 자동 전환"과
        # 같은 가드를 스트리밍 경로에도 둔다 - 2026-08 통합 병합으로 프론트가
        # 이제 이 함수(/decide/stream)를 기본 검색 경로로 쓰므로, 이 가드가
        # 없으면 로컬에서 LLM 키 없이 실행할 때 run_single_debate_stream이
        # adk_pipeline을 통해 바로 실패한다 - run_danawa_only_debate_stream()으로
        # 대체해 LLM 키 없이도(다나와 실측 가격만으로) 계속 동작하게 한다.
        async for event in run_danawa_only_debate_stream(query):
            yield event
        return
    if needs_clarification(query):
        yield {"type": "final", "result": (await run_clarify(query)).model_dump()}
        return

    async for event in run_single_debate_stream(query):
        yield event


_MAX_CLARIFY_ROUNDS = 2
# facet 중 2개 이상이 이미 질의 텍스트에 반영됐으면(=사용자가 Human-in-the-loop을
# 2라운드 이상 거쳤으면) 더 묻지 않고 바로 propose/challenge/judge 파이프라인으로
# 넘긴다. 라운드마다 검색 결과가 바뀌면서 이전엔 안 갈리던 축이 뒤늦게 애매해지는
# 경우가 있는데, 이걸 그대로 다 물어보면 사용자는 몇 라운드나 답했는데도 계속 새
# 선택 화면이 뜨는 것처럼 느껴진다 — 남은 후보를 좁히는 건 그 시점부턴 judge가
# 근거(그라운딩/검증)로 판단하게 맡긴다.


async def _search_danawa_urls_with_fallback(
    query: str, base_query: str | None
) -> tuple[list[str], str]:
    """가격 실측용 다나와 URL 검색(사용자 요청, 2026-08-14: "이런식으로 가격
    정보를 찾지 못하는 결과는 없어야해"). AI 상세검색으로 facet을 여러 개
    이어붙인 아주 구체적인 검색어(예: "초코파이 오리온 초코파이 바나나 468g")는
    다나와 검색엔진이 토큰이 많거나 표현이 어색해 결과가 아예 안 나올 수 있다 -
    실제 상품이 없어서가 아니라 검색어 자체의 문제일 수 있다는 뜻이다. query가
    빈손이면 그 조합을 만든 원래(더 넓은) base_query로 한 번 더 시도한다.
    (effective_query, urls) 대신 (urls, effective_query) 순서로 반환 - 호출자가
    이후 _finalize_danawa_only에 "실제로 무엇으로 검색해서 찾았는지"를 넘겨야
    해서다."""
    urls = await price_table_module._search_danawa_urls(query, limit=price_table_module.DANAWA_ONLY_SEARCH_LIMIT)
    if urls:
        return urls, query
    if base_query and base_query.strip() and base_query.strip() != query.strip():
        fallback_urls = await price_table_module._search_danawa_urls(
            base_query, limit=price_table_module.DANAWA_ONLY_SEARCH_LIMIT
        )
        if fallback_urls:
            logger.info(
                "danawa-only fallback: %r found nothing, retried with base_query %r", query, base_query
            )
            return fallback_urls, base_query
    return [], query


async def run_danawa_only_debate(
    query: str, base_query: str | None = None
) -> DecideResponse | BulkDecideResponse:
    """LLM API 비용 절감을 위한 임시 로컬 실험 경로 - gpt/groq/deepseek
    제안도, judge 최종 결정도 전부 건너뛴다. LLM 호출 0번. 다나와 실측
    가격표(다나와 직접검색만)에서 A등급(구매 링크 생성 가능) offer를
    규칙 기반으로 최종 추천으로 쓴다.

    run_debate()에서 자동으로 타지 않는다 - /decide/danawa-only 전용
    엔드포인트에서만 명시적으로 호출한다. 다나와 데이터가 아예 없거나
    A등급 offer가 없으면(=구매 링크를 만들 수 없으면) RuntimeError를 던진다 -
    "링크 없는 추천은 만들지 않는다" 원칙은 LLM 없이도 동일하게 지킨다.

    속도 최적화(사용자 요청, 두 번째 라운드) - 캐시 안 된 쿼리를 실측해보니
    Tavily 검색 자체가 15~20초로 압도적 병목이었다(다나와 직접검색은
    2~4초, 상세페이지 5건 페치는 3~4초). 이 경로는 LLM 에이전트가 없어서
    Tavily 결과가 다른 어디에도 안 쓰이므로, Tavily를 아예 호출하지
    않고 search_danawa()만으로 pcode를 찾는다 - 후보 수는 줄지만(합집합의
    Tavily 쪽 절반을 잃음) 응답은 몇 배 빨라진다. run_single_debate()(LLM
    경로)는 Tavily 결과를 LLM 에이전트와도 공유해야 해서 그대로 둔다.

    두 갈래(사용자 요청, 세 번째 라운드, 2026-08-11):
    1) 후보들이 같은 상품의 스펙 변형(색상/용량)이면 - 예: "맥북에어 m2" -
       하나를 고르되 pick_primary()(offer가 가장 많은 "풍부한" 페이지)가
       아니라 cheapest_across_tables()(A등급 절대 최저가)로 고른다.
       실측에서 관련도 1위 변형이 이미 가장 쌌는데도(1,307,990원) 풍부함
       기준 때문에 1,669,990원짜리가 나갔던 문제를 이렇게 고친다.
    2) 후보들이 브랜드/모델 자체가 다른 상품들이면(_is_single_product_family
       False) - 예: "노트북" - 하나를 억지로 고르지 않고 BulkDecideResponse
       (기존 '대량구매' 응답과 동일 스키마, 프론트도 그대로 재사용)로 후보를
       가격 오름차순으로 나열해 사용자가 직접 고르게 한다.

    후보 수는 DANAWA_ONLY_SEARCH_LIMIT(3) - 사용자 요청(네 번째 라운드,
    2026-08-11: "한번에 5개 찾아주는거 너무 느린데")로 5에서 줄였다.

    base_query(2026-08-14, "이런식으로 가격 정보를 찾지 못하는 결과는 없어야해") -
    AI 상세검색으로 facet을 여러 개 이어붙인 아주 구체적인 검색어(예: "초코파이
    오리온 초코파이 바나나 468g")는 다나와 검색엔진이 토큰이 많거나 어색해서
    결과가 아예 안 나올 수 있다 - 그러면 그 조합을 만든 원래(더 넓은) 검색어로
    한 번 더 시도한다(_search_danawa_urls_with_fallback 참고)."""
    urls, effective_query = await _search_danawa_urls_with_fallback(query, base_query)
    danawa_tables = await price_table_module.fetch_price_tables_for_urls(urls)
    return await _finalize_danawa_only(effective_query, danawa_tables, original_query=query)


async def run_danawa_only_debate_stream(
    query: str, base_query: str | None = None
) -> AsyncIterator[dict[str, Any]]:
    """run_danawa_only_debate()의 스트리밍 버전 - 사용자 요청(네 번째 라운드,
    2026-08-11: "1개 서치 완료되면 1개 올려줘 먼저"). 후보 3개를 asyncio.gather로
    한꺼번에 기다리는 대신 stream_price_tables_for_urls()로 끝나는 대로 하나씩
    {"type": "candidate", ...} 이벤트로 내보내고, 전부(또는 실패한 나머지를
    빼고) 모이면 run_danawa_only_debate()와 완전히 동일한 로직(_finalize_danawa_only)
    으로 최종 판단을 만들어 {"type": "final", "result": ...}로 내보낸다.

    /decide/danawa-only/stream 전용 - run_debate()에서 자동으로 타지 않는다.

    base_query(2026-08-14) - run_danawa_only_debate() 참고. 여기서도 query가
    아무 결과도 못 찾으면 base_query로 한 번 더 시도해 진짜 빈손으로 끝나는
    경우를 최대한 줄인다."""
    urls, effective_query = await _search_danawa_urls_with_fallback(query, base_query)
    if not urls:
        yield {"type": "error", "message": f"다나와에서 '{query}'에 대한 가격 정보를 찾지 못했다(검색/실측 모두 실패)."}
        return

    danawa_tables: list[tuple[Any, danawa.DanawaResult]] = []
    async for table, raw in price_table_module.stream_price_tables_for_urls(urls):
        danawa_tables.append((table, raw))
        offer = price_table_module.cheapest_linkable_raw_offer(raw)
        # candidate 이벤트에는 구매 URL을 안 채운다(사용자 요청, 2026-08-11:
        # "검색시간을 더 줄일수있나") - resolve_purchase_url()은 2홉 네트워크
        # 요청이고, 그마저도 prod.danawa.com 상세페이지 페치와 같은 도메인
        # 스로틀(0.5초)을 공유해 서로 대기열에 줄을 선다. 어차피 최종 확정
        # 단계(_finalize_danawa_only)에서 필요한 offer(들)만 다시 resolve하므로,
        # 여기서 3개 전부 미리 resolve하면 최종 당첨자 몫은 완전히 중복이고
        # 나머지는 아예 버려지는 순수 낭비였다 - 프론트도 로딩 중 후보 목록은
        # 텍스트로만 보여줄 뿐 클릭 링크로 안 쓴다.
        yield {
            "type": "candidate",
            "product_name": table.product_name,
            "price": f"{offer['price_krw']:,}원" if offer is not None else None,
            "retailer": offer["seller"] if offer is not None else None,
            "url": None,
        }

    try:
        result = await _finalize_danawa_only(effective_query, danawa_tables, original_query=query)
    except RuntimeError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    yield {"type": "final", "result": result.model_dump()}


async def _finalize_danawa_only(
    query: str,
    danawa_tables: list[tuple[Any, danawa.DanawaResult]],
    *,
    original_query: str | None = None,
) -> DecideResponse | BulkDecideResponse:
    """run_danawa_only_debate()/run_danawa_only_debate_stream() 공유 로직 -
    이미 페치된 danawa_tables로부터 최종 DecideResponse(단일 상품) 또는
    BulkDecideResponse(애매모호한 검색어 후보 목록)를 만든다.

    original_query(2026-08-14) - query가 _search_danawa_urls_with_fallback으로
    base_query로 대체된 경우("초코파이 오리온 초코파이 바나나 468g"를 못 찾아
    "초코파이"로 넓혀 찾은 경우), 사용자가 원래 물어본 게 뭐였는지 reasoning에
    솔직하게 남긴다 - 조용히 다른 상품을 최종 추천으로 내밀지 않는다."""
    used_fallback = bool(original_query) and original_query.strip() != query.strip()
    fallback_note = (
        f"'{original_query}'에 정확히 맞는 상품을 찾지 못해 더 넓은 검색어 '{query}'로 찾은 결과입니다. "
        if used_fallback
        else ""
    )

    if not danawa_tables:
        raise RuntimeError(f"다나와에서 '{query}'에 대한 가격 정보를 찾지 못했다(검색/실측 모두 실패).")

    if not price_table_module._is_single_product_family(danawa_tables):
        options = await price_table_module.build_ambiguous_options(danawa_tables)
        if not options:
            raise RuntimeError(f"'{query}'는 여러 상품에 걸친 검색어인데, 구매 링크를 만들 수 있는 후보가 없다.")
        return BulkDecideResponse(
            query=original_query or query,
            proposals=[],
            decision=BulkDecision(
                options=options,
                reasoning=(
                    f"{fallback_note}'{query}'는 서로 다른 상품에 걸친 검색어라 하나로 단정하지 않고, "
                    f"실측 최저가 순으로 후보 {len(options)}개를 보여드립니다."
                ),
            ),
            price_range=_compute_price_range(options),
        )

    best = price_table_module.cheapest_across_tables(danawa_tables)
    if best is None:
        raise RuntimeError(f"'{query}' 가격표는 있지만 구매 링크를 만들 수 있는(A등급) 판매처가 없다.")
    table, raw_result, offer = best

    resolved_url = await price_table_module.resolve_purchase_url(offer)
    if resolved_url is None:
        raise RuntimeError(f"'{table.product_name}' 최저가 판매처({offer['seller']}) 구매 링크 해석에 실패했다.")

    decision = Decision(
        product_name=table.product_name or query,
        price=f"{offer['price_krw']:,}원",
        retailer=offer["seller"],
        url=resolved_url,
        reasoning=f"{fallback_note}다나와 실측 최저가(A등급, 구매링크 검증됨) - LLM 미사용, 규칙 기반 선택",
        chosen_agent="danawa",
        price_source="danawa_offer",
    )
    decision = await price_table_module.exclude_price_comparison_site_as_final_pick(decision, [], danawa_tables)

    return DecideResponse(query=original_query or query, proposals=[], decision=decision, price_table=table)


async def run_elevenst_only_debate(query: str) -> DecideResponse:
    """다나와가 아니라 11번가 오픈 API(ProductSearch)만으로 검색하는 실험
    경로(2026-08-20, "다나와 리서치 말고 11번가 API만 사용해서 검색하는걸로
    바꿔서 한번 띄워줘봐") - run_danawa_only_debate()와 같은 "LLM 미사용,
    구조화 데이터만" 원칙을 따르지만 소스가 완전히 다르다(스크래핑이 아니라
    11번가 1st-party API).

    _DanawaFetchNode가 pick_primary()에 관련성 가드 없이 썼다가 무관한 상품이
    골라진 회귀(adk_pipeline.py 주석 참고)를 반복하지 않기 위해, 여기서도
    검색 결과 각각에 _product_name_matches(query, product_name)를 적용해
    질의와 실제로 맞는 상품만 후보로 남긴 뒤 그중 최저가를 고른다 - API
    자체의 정렬(sortCd=A)이 실측상 항상 가격 오름차순은 아니었어서
    (2026-08-20 실측: 정렬 안 됨 확인) 클라이언트에서 직접 최저가를 고른다."""
    items = await elevenst.search_elevenst(query, limit=10)
    relevant = [it for it in items if price_table_module._product_name_matches(query, it["product_name"])]
    if not relevant:
        raise RuntimeError(f"11번가에서 '{query}'에 대해 관련성 있는 상품을 찾지 못했다.")

    best = min(relevant, key=lambda it: it["price_krw"])
    decision = Decision(
        product_name=best["product_name"],
        price=f"{best['price_krw']:,}원",
        retailer=best["seller"],
        url=best["url"],
        reasoning="11번가 오픈 API(ProductSearch) 실측 - LLM 미사용, 구조화 데이터 기반 최저가 선택",
        chosen_agent="elevenst",
        price_source="elevenst_offer",
    )
    proposals = [
        Proposal(
            agent="elevenst",
            product_name=it["product_name"],
            price=f"{it['price_krw']:,}원",
            retailer=it["seller"],
            url=it["url"],
            reasoning="11번가 오픈 API 검색 결과",
            verified=True,
        )
        for it in relevant
    ]
    return DecideResponse(query=query, proposals=proposals, decision=decision)


async def run_elevenst_only_debate_stream(query: str) -> AsyncIterator[dict[str, Any]]:
    """run_elevenst_only_debate()의 스트리밍 버전(2026-08-20) - "다나와껀 빼고
    11번가 API만 붙여줘" 요청으로 메인 검색 흐름(/decide/stream)이 이 경로를
    타게 됐다. run_debate_stream()과 이벤트 모양을 맞춰야 프론트가 그대로
    재사용된다 - status 이벤트 하나(진행 표시용) 후 final 이벤트 하나."""
    yield {"type": "status", "stage": "searching"}
    try:
        result = await run_elevenst_only_debate(query)
    except RuntimeError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    yield {"type": "final", "result": result.model_dump()}


# check_clarify_facets()의 base_query 재사용 필터링 전용(사용자 요청, 2026-08-13:
# "조금 더 빠르게 검색기능이 되면 좋겠어"). 필터링 결과가 이보다 적으면 표본이
# 너무 좁아 facet 품질이 나빠질 수 있으니, 필터링을 포기하고 base_query의
# 넓은 표본을 그대로 쓴다(추가 검색은 하지 않는다 - 속도가 이 최적화의 목적이라
# 여기서 또 search.danawa.com을 때리면 본전도 못 찾는다).
MIN_FILTERED_CLARIFY_ITEMS = 3

# _enrich_facets_per_brand가 브랜드당 병렬 DeepSeek 호출을 최대 몇 개까지
# 내보낼지(토큰 절약, 2026-08-19) - brand_facet.options는 이미 인기순이라
# 상위 몇 개만 보강해도 실사용 대부분을 커버한다. 표시되는 브랜드 목록
# 자체(최대 MAX_BRAND_OPTIONS=15)는 그대로고, 이 값은 그중 몇 개까지
# "다른 축도 더 채워줄지"만 정한다.
_MAX_BRAND_ENRICH_FANOUT = 6


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _filter_items_by_extra_terms(
    items: list[danawa_search.DanawaSearchItem], query: str, base_query: str
) -> list[danawa_search.DanawaSearchItem]:
    """base_query로 얻은(캐시 재사용) 넓은 표본을, 사용자가 그 뒤에 덧붙인
    단어들(예: base_query="핸드폰", query="핸드폰 삼성전자"의 "삼성전자")로
    상품명을 걸러 좁힌다 - 네트워크 요청 없이 순수 로컬 필터링."""
    base_tokens = {_normalize_for_match(t) for t in base_query.split()}
    extra_tokens = [
        _normalize_for_match(t) for t in query.split() if _normalize_for_match(t) not in base_tokens
    ]
    if not extra_tokens:
        return items
    filtered = [
        item
        for item in items
        if all(term in _normalize_for_match(item["product_name"]) for term in extra_tokens)
    ]
    return filtered if len(filtered) >= MIN_FILTERED_CLARIFY_ITEMS else items


def _items_for_brand(brand: str, names: list[str]) -> list[str]:
    nb = _normalize_for_match(brand)
    return [name for name in names if nb in _normalize_for_match(name)]


async def _enrich_facets_per_brand(
    facets: list[ClarifyFacet], names: list[str], query: str
) -> list[ClarifyFacet]:
    """브랜드가 여러 개 섞인 채로 한 번에 facet을 뽑으면, DeepSeek이 각 facet마다
    총 MAX_OPTIONS_PER_FACET(6)개 안에서 모든 브랜드가 경쟁한다 - 다나와 검색
    결과가 특정 브랜드로 치우친 카테고리(예: "핸드폰"의 삼성전자)면, 소수 브랜드
    (APPLE)의 시리즈가 인기순 정렬에서 밀려 상위 6개에 아예 못 들 수 있다
    (사용자 요청, 2026-08-13: "APLLE 을 선택했을때 시리즈 후보가 너무 적어 ...
    다른 질문을 했을때도 이런문제가 없으면"). 그래서 브랜드별로 그 브랜드
    상품명만 모아 별도로(각자 자기 몫의 6개 예산을 온전히 받아) 다시 facet을
    뽑고(asyncio.gather로 병렬), 같은 라벨의 facet에 새 옵션만 합쳐 넣는다.
    라벨을 다시 자유롭게 고르게 두면 같은 개념도 호출마다 "시리즈"/"모델"처럼
    다르게 이름 붙어 병합이 안 되므로, required_labels로 원래 라벨을 그대로 쓰라고
    강제한다(deepseek.extract_facets_from_names 참고). 그래도 안 맞으면(모델이
    지시를 어기면) 그 라벨은 그냥 원래 결과 그대로 둔다 - 실패해도 기존 동작보다
    나빠지지 않는다.

    names는 상품명 문자열 목록이면 출처는 무관하다(2026-08-16, _extract_facets
    통합 - 다나와 직접 검색 결과든 Tavily 검색 결과 제목이든 상관없음).

    토큰 절약(2026-08-19) - brand_facet.options는 이미 extract_facets_from_names가
    인기순으로 정렬해뒀다(브랜드는 _WIDE_CAP_LABEL_PATTERN이라 최대
    MAX_BRAND_OPTIONS=15개까지 남아있을 수 있음). 15개 전부를 병렬 DeepSeek
    호출로 보강하면 이 함수 하나가 요청 한 번에 최대 15번 DeepSeek를 부른다 -
    실제로 이득을 보는 쪽은 상위 소수 브랜드다(사용자가 실제로 고르는 것도
    거의 항상 인기 브랜드). 상위 _MAX_BRAND_ENRICH_FANOUT개만 보강하고 나머지는
    (전체 표본 기준 인기순 정렬 결과를) 그대로 둔다 - 표시되는 브랜드 개수 자체는
    그대로고, 인기 브랜드 몇 개만 시리즈/모델 같은 다른 축이 더 풍부해진다."""
    brand_facet = next((f for f in facets if deepseek._BRAND_LABEL_PATTERN.search(f.label)), None)
    if brand_facet is None or len(brand_facet.options) < 2:
        return facets

    other_labels = [f.label for f in facets if f is not brand_facet]
    if not other_labels:
        return facets

    brands_to_enrich = brand_facet.options[:_MAX_BRAND_ENRICH_FANOUT]
    try:
        per_brand_facets = await asyncio.gather(
            *(
                deepseek.extract_facets_from_names(
                    query, _items_for_brand(brand, names), required_labels=other_labels
                )
                for brand in brands_to_enrich
            )
        )
    except Exception:
        logger.exception("check_clarify_facets: 브랜드별 facet 보강 실패, 원래 결과 그대로 사용")
        return facets

    enriched = []
    for facet in facets:
        if facet is brand_facet:
            enriched.append(facet)
            continue
        merged_options = list(facet.options)
        seen = {_normalize_for_match(o) for o in merged_options}
        for brand_facets in per_brand_facets:
            match = next((f for f in brand_facets if f.label == facet.label), None)
            if match is None:
                continue
            for option in match.options:
                key = _normalize_for_match(option)
                if key not in seen:
                    seen.add(key)
                    merged_options.append(option)
        if merged_options != facet.options:
            enriched.append(facet.model_copy(update={"options": merged_options}))
        else:
            enriched.append(facet)
    return enriched


_DEVICE_ECOSYSTEM_TERMS = ["갤럭시", "아이폰"]
# 이보다 표본이 적으면 그 생태계는 다나와 보충 검색을 한 번 더 돌린다(아래
# _ecosystem_name_pool) - 실측(2026-08-14, 사용자 리포트 "갤럭시랑 아이폰이랑
# 비슷한 비율로 기종이 뜨게 하고 싶었어" 조사 중 발견): "핸드폰 케이스" 다나와
# 검색 결과 40개 중 갤럭시가 36개, 아이폰이 1개뿐이었다 - facet 추출/균형
# 로직이 완벽해도 원본 표본에 아이폰 매물 자체가 거의 없으면 소용없다.
_DEVICE_ECOSYSTEM_SUPPLEMENT_MIN_ITEMS = 3


def _items_for_substring(term: str, names: list[str]) -> list[str]:
    nt = _normalize_for_match(term)
    return [name for name in names if nt in _normalize_for_match(name)]


async def _ecosystem_name_pool(eco: str, names: list[str], query: str) -> list[str]:
    """이 생태계(갤럭시/아이폰) 상품명이 기존 표본에 부족하면, 그 생태계 이름을
    검색어 앞에 붙여 다나와에 보충 검색을 한 번 더 돌려 실제 매물을 채운다.
    이미 충분하면(>= _DEVICE_ECOSYSTEM_SUPPLEMENT_MIN_ITEMS) 추가 네트워크
    요청 없이 기존 표본만 쓴다 - 매번 보충 검색을 돌리면 다나와 Crawl-delay
    비용이 그 생태계가 이미 충분한 흔한 경우에도 매번 붙는다. 다나와 검색
    자체가 실패해도(네트워크 오류 등) 기존 표본만으로 계속 진행한다."""
    pool = _items_for_substring(eco, names)
    if len(pool) >= _DEVICE_ECOSYSTEM_SUPPLEMENT_MIN_ITEMS:
        return pool
    try:
        items = await price_table_module._search_danawa_items(
            f"{eco} {query}", limit=price_table_module.CLARIFY_SEARCH_LIMIT
        )
    except Exception:
        logger.exception("check_clarify_facets: %s 보충 검색 실패, 기존 표본만 사용", eco)
        return pool
    seen = {_normalize_for_match(n) for n in pool}
    merged = list(pool)
    for item in items:
        name = item["product_name"]
        key = _normalize_for_match(name)
        if key not in seen:
            seen.add(key)
            merged.append(name)
    return merged


async def _enrich_device_models_by_ecosystem(
    facets: list[ClarifyFacet], names: list[str], query: str
) -> list[ClarifyFacet]:
    """'핸드폰 기종' facet이 특정 기기 브랜드로 쏠리는 문제(사용자 리포트,
    2026-08-14: "갤럭시랑 아이폰이랑 비슷한 비율로 기종이 뜨게 하고 싶었어") -
    _enrich_facets_per_brand는 "케이스" 브랜드(신지모루/슈피겐 등)별로 예산을
    나눠주는데, 그 케이스 브랜드 자체가 갤럭시 위주로 팔면 소용이 없다(케이스
    브랜드 축과 기기 브랜드 축은 서로 다른 문제). 기기 생태계(갤럭시/아이폰)
    문자열로 상품명 표본을 나누고(부족하면 보충 검색까지 돌려) 각자 자기 몫의
    예산으로 다시 '핸드폰 기종'만 추출한 뒤, 새로 찾은 값을 합쳐
    deepseek._balance_device_models_by_ecosystem로 다시 균형 있게 자른다."""
    device_facet = next((f for f in facets if f.label == deepseek._DEVICE_MODEL_LABEL), None)
    if device_facet is None:
        return facets

    try:
        ecosystem_pools = await asyncio.gather(
            *(_ecosystem_name_pool(eco, names, query) for eco in _DEVICE_ECOSYSTEM_TERMS)
        )
        per_ecosystem = await asyncio.gather(
            *(
                deepseek.extract_facets_from_names(
                    query, pool, required_labels=[deepseek._DEVICE_MODEL_LABEL]
                )
                for pool in ecosystem_pools
            )
        )
    except Exception:
        logger.exception("check_clarify_facets: 기종 생태계별 facet 보강 실패, 원래 결과 그대로 사용")
        return facets

    merged_options = list(device_facet.options)
    seen = {_normalize_for_match(o) for o in merged_options}
    for eco_facets in per_ecosystem:
        match = next((f for f in eco_facets if f.label == deepseek._DEVICE_MODEL_LABEL), None)
        if match is None:
            continue
        for option in match.options:
            key = _normalize_for_match(option)
            if key not in seen:
                seen.add(key)
                merged_options.append(option)

    if merged_options == device_facet.options:
        return facets

    # 인기순 집계도 보충 검색으로 늘어난 표본을 반영해야, 보충 검색으로 새로
    # 찾은 기종이 전부 count=0으로 묶여 정렬이 무의미해지지 않는다.
    popularity_pool = names + [n for pool in ecosystem_pools for n in pool]
    balanced = deepseek._balance_device_models_by_ecosystem(
        merged_options, popularity_pool, deepseek.MAX_BRAND_OPTIONS
    )
    return [f.model_copy(update={"options": balanced}) if f is device_facet else f for f in facets]


def _build_facet_value_incidence(facets: list[ClarifyFacet], names: list[str]) -> dict[str, set[int]]:
    """상품명 하나는 브랜드·시리즈·용량 등 여러 facet 값을 동시에 잇는
    하이퍼엣지다 - 이 dict(facet 값(정규화) -> 그 값이 등장하는 상품명 인덱스
    집합)가 곧 그 (facet 값 정점) x (상품 하이퍼엣지) incidence 구조다.
    _attach_facet_crossfilter/_facet_centrality가 공유하는 빌더 - 두 값이
    같은 상품에 같이 등장하는지(=incidence 집합의 교집합이 있는지) 판정하는
    데 쓰인다."""
    normalized_names = [_normalize_for_match(name) for name in names]
    incidence: dict[str, set[int]] = {}
    for facet in facets:
        for option in facet.options:
            key = _normalize_for_match(option)
            if key not in incidence:
                incidence[key] = {i for i, n in enumerate(normalized_names) if key in n}
    return incidence


def _attach_facet_crossfilter(facets: list[ClarifyFacet], names: list[str]) -> list[ClarifyFacet]:
    """facet 쌍 전부(브랜드 한정이 아니다) 사이의 연관을 상품명에서 직접 계산해
    붙인다(사용자 요청, 2026-08-13: "삼성전자를 누르면 시리즈에 삼성전자에 관한것만"
    -> 2026-08-14: "시리즈에 초코파이 바나나를 골랐다면 용량에 없는것들은
    선택할수 없게" - 브랜드 특수 케이스였던 걸 모든 facet 쌍으로 일반화했다).
    검색을 다시 하지 않고, 이미 받아온 names(상품명)만으로 "이 두 옵션이 같은
    상품명에 같이 등장하는가"를 모든 (선택 가능한 facet, 대상 facet) 쌍에 대해
    계산한다 - 그래서 프론트가 어느 facet에서든 값을 고르는 순간(그 자체로는
    아직 검색을 트리거하지 않는다) 아직 안 고른 다른 facet들의 보이는 옵션만
    즉시 좁혀 보여줄 수 있다(여러 개를 고르면 교집합은 프론트가 계산한다).

    2026-08-16, 하이퍼그래프 incidence 구조로 재구성 - "같은 상품명에 같이
    등장하는가"를 매번 전체 상품명을 재스캔해 판정하는 대신(브루트포스
    O(facet² x 선택지 x 선택지 x 상품명)), _build_facet_value_incidence로
    한 번만 만든 값→상품 인덱스 집합의 교집합 유무로 판정한다 - 두 판정은
    수학적으로 동치(어떤 상품명 i가 두 값을 모두 포함 ⇔ 두 값의 incidence
    집합이 겹침)라 결과는 그대로다."""
    if len(facets) < 2:
        return facets

    incidence = _build_facet_value_incidence(facets, names)

    def _relevant(selector_value: str, target_options: list[str]) -> list[str]:
        selector_idx = incidence.get(_normalize_for_match(selector_value), set())
        if not selector_idx:
            return []
        return [
            option
            for option in target_options
            if incidence.get(_normalize_for_match(option), set()) & selector_idx
        ]

    updated = []
    for target in facets:
        by_selection: dict[str, list[str]] = {}
        for selector in facets:
            if selector is target:
                continue
            for value in selector.options:
                relevant = _relevant(value, target.options)
                # 이 선택이 target의 옵션을 실제로 좁혀줄 때만(전체가 그대로
                # 나오거나 아예 안 나오면 매핑을 안 붙인다) 쓸모없는 데이터를
                # 응답에 얹지 않는다.
                if relevant and len(relevant) < len(target.options):
                    by_selection[value] = relevant
        updated.append(target.model_copy(update={"options_by_selection": by_selection}) if by_selection else target)
    return updated


# 상세검색 facet을 넓은(거시적) 기준부터 좁은(미시적) 기준 순서로 보여주기 위한
# 우선순위 힌트(사용자 요청, 2026-08-14: "거시적인 선택에서 미시적인 선택으로
# 점차 줄여나가게"). 라벨에 이 키워드가 포함되면 그 순번을 쓴다 - 못 찾은
# 라벨은 _facet_sort_key가 incidence 기반 중심성으로 다시 정렬한다(아래).
# 실제 좁히기 자체는 _attach_facet_crossfilter가 순서와 무관하게 다
# 계산해두므로, 이건 어떤 순서로 "물어보면" 자연스러운지에 대한 화면 표시
# 순서일 뿐이다.
_FACET_ORDER_HINTS = [
    # "기종"(핸드폰 기종/호환기종)이 맨 앞 - 사용자 요청(2026-08-14: "검색
    # 순서에서 핸드폰 기종이 가장 먼저 위로 올라가야할 것 같은데"). 액세서리
    # 검색에서는 "내 기기에 맞는지"가 카테고리·브랜드보다 더 먼저 정해야 하는
    # 기준이라는 판단.
    "기종",
    "카테고리", "브랜드", "제조사", "시리즈", "모델", "타입", "종류",
    "용량", "무게", "사이즈", "용기형태", "구매유형", "특징", "색상",
]


def _facet_display_order(facet: ClarifyFacet) -> int:
    for i, hint in enumerate(_FACET_ORDER_HINTS):
        if hint in facet.label:
            return i
    return len(_FACET_ORDER_HINTS)


def _facet_centrality(facet: ClarifyFacet, incidence: dict[str, set[int]]) -> float:
    """이 facet의 선택지들이 다른 상품들과 평균적으로 얼마나 폭넓게
    공존하는가(incidence 그래프에서의 평균 degree) - 넓은(거시적) 축일수록
    선택지 하나하나가 더 많은 상품에 걸쳐 등장하는 경향이 있다는 근사다.
    _facet_sort_key가 _FACET_ORDER_HINTS로 못 잡은 facet들 사이의
    타이브레이커로만 쓴다."""
    if not facet.options:
        return 0.0
    return sum(len(incidence.get(_normalize_for_match(o), set())) for o in facet.options) / len(facet.options)


def _facet_sort_key(facet: ClarifyFacet, incidence: dict[str, set[int]]) -> tuple[int, float]:
    """_facet_display_order(힌트 기반)가 우선이고, 힌트가 못 잡은 facet들
    사이에서만 incidence 중심성(내림차순)으로 다시 가른다 - 힌트가 이미
    잡은 facet은 중심성을 아예 안 보므로(표본이 작을 때 중심성 신호가
    약해지는 문제로부터 안전) 기존 정렬 결과가 그대로 보존된다."""
    order = _facet_display_order(facet)
    if order != len(_FACET_ORDER_HINTS):
        return (order, 0.0)
    return (order, -_facet_centrality(facet, incidence))


def _apply_persona_ordering(facets: list[ClarifyFacet], persona: dict[str, str]) -> list[ClarifyFacet]:
    """사용자 페르소나(2026-08-15, "냉장고 살 때랑 콜라 살 때 쓰는 메타데이터가
    다르다" - 카테고리별 facet 자체는 이미 실제 검색 결과에서 즉석에 뽑히므로
    해결됐고, 이건 그 위에서 "이 사용자가 이 라벨에서 평소/이번 세션에 어떤 값을
    골랐는가"를 반영하는 것). persona는 {facet 라벨: 선호 값} - 세션(이번 대화
    누적)과 로그인 계정 영구 기록(app.preferences)을 프론트/main.py에서 이미
    병합해 넘긴다. 하드 필터가 아니라 그 값이 실제로 이 facet의 옵션 목록에
    있을 때만 맨 앞으로 당기는 소프트 정렬이다 - 옵션 자체를 줄이거나 없는
    값을 지어내 추가하지 않는다."""
    if not persona:
        return facets
    reordered = []
    for facet in facets:
        preferred = persona.get(facet.label)
        if preferred and preferred in facet.options:
            options = [preferred] + [o for o in facet.options if o != preferred]
            reordered.append(facet.model_copy(update={"options": options}))
        else:
            reordered.append(facet)
    return reordered


async def _extract_facets(
    query: str, names: list[str], persona: dict[str, str] | None = None
) -> list[ClarifyFacet]:
    """상품명 목록(다나와 직접 검색이든 Tavily 검색 결과 제목이든, 출처
    무관)에서 facet을 뽑는 공유 파이프라인(2026-08-16, README "장기 과제"
    후속) - check_clarify_facets(다나와 직접 검색)와 adk_pipeline의 내부
    안전망(_extract_clarify_options, Tavily 결과)이 이전엔 서로 다른 추출
    로직(동적 facet vs 고정 4축)을 썼는데, 프론트가 이미 facet 렌더링을
    완전히 지원해서 별도 UI 변경 없이 하나로 합칠 수 있었다."""
    facets = await deepseek.extract_facets_from_names(query, names)
    facets = await _enrich_facets_per_brand(facets, names, query)
    facets = await _enrich_device_models_by_ecosystem(facets, names, query)
    facets = _attach_facet_crossfilter(facets, names)
    incidence = _build_facet_value_incidence(facets, names)
    facets = sorted(facets, key=lambda f: _facet_sort_key(f, incidence))
    facets = _apply_persona_ordering(facets, persona or {})
    return facets


def _normalize_for_query_match(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _strip_query_answered_options(query: str, facets: list[ClarifyFacet]) -> list[ClarifyFacet]:
    """사용자가 검색어에 이미 쓴 단어를 facet 선택지로 또 보여주는 문제(2026-08
    사용자 리포트 "스탠리 텀블러 검색했는데 물어보는 게 반복되고 많다") - "텀블러"를
    검색했는데 "제품분류" facet이 선택지로 "텀블러"를 또 보여주는 식이라 이미 답한
    걸 다시 묻는 것처럼 느껴진다. FACET_CLARIFY_INSTRUCTIONS에 이미 답한 개념은
    값으로 넣지 말라고 지시했지만 DeepSeek이 안정적으로 안 지켜(실측: "스탠리
    텀블러"/"나이키 반팔티" 둘 다 재현) 여기서 한 번 더 거른다. 질의를 공백 제거
    + 소문자로 정규화해 그 안에 그대로 부분 문자열로 포함되는 옵션만 제거한다 -
    "반팔티"처럼 질의에 있는 그대로의 표현만 잡고, "반팔 티셔츠"처럼 표현이 달라진
    동의어까지는 못 잡는다(그건 프롬프트 쪽 개선 영역으로 남겨둔다)."""
    normalized_query = _normalize_for_query_match(query)
    result: list[ClarifyFacet] = []
    for facet in facets:
        kept = [
            opt
            for opt in facet.options
            if _normalize_for_query_match(opt) not in normalized_query
        ]
        if len(kept) == len(facet.options):
            # 아무것도 안 걸러졌으면 원래 facet을 그대로 둔다 - 옵션이 원래부터
            # 1개뿐인 경우까지 이 필터가 건드릴 이유는 없다(그건 이 함수의 책임이
            # 아니라 추출 쪽 문제).
            result.append(facet)
            continue
        if len(dict.fromkeys(kept)) < 2:
            continue
        options_by_selection = None
        if facet.options_by_selection:
            filtered = {
                selector: [v for v in values if v in kept]
                for selector, values in facet.options_by_selection.items()
            }
            options_by_selection = {k: v for k, v in filtered.items() if v} or None
        result.append(
            ClarifyFacet(label=facet.label, options=kept, options_by_selection=options_by_selection)
        )
    return result


async def check_clarify_facets(
    query: str, base_query: str | None = None, persona: dict[str, str] | None = None
) -> ClarifyResponse:
    """AI 상세검색(2026-08-12 요청) - "음료수"처럼 짧고 애매한 검색어를 다나와
    실제 검색 결과 상품명에 근거해 몇 가지 기준(facet)으로 좁혀나가도록 DeepSeek에게
    물어본다(원래 Qwen으로 붙였다가, Model Studio 계정의 과금 플랜 활성화 문제로
    이미 키가 있고 바로 되는 DeepSeek로 옮겼다). run_danawa_only_debate()/
    run_danawa_only_debate_stream()과는 완전히 분리된 별도 진입점이다 - 그 둘은
    "LLM 호출 0번"이 테스트로 고정된 불변식이라(test_run_danawa_only_debate_never_calls_any_llm)
    여기서 DeepSeek를 부르는 로직을 거기 안에 섞으면 안 된다. 프론트가 이 함수를
    먼저(짧은 쿼리에 한해) 호출해보고, facets가 비어 있으면(=명확한 검색어이거나
    DeepSeek 호출 실패) 그대로 danawa-only 빠른 경로로 진행한다.

    needs_clarification()이 False면 검색조차 하지 않고 즉시 빈 결과를 반환한다 -
    대부분의(구체적인) 검색어는 이 함수를 호출해도 search.danawa.com 요청도,
    DeepSeek 호출도 전혀 없이 즉시 끝난다.

    "하이"처럼 상품과 무관한 인사말/잡담도 같은 이유로 즉시 빈 결과를 반환한다
    (사용자 요청, 2026-08-15: "상품으로 인식못하는 말을 들으면 처리해야하는
    속도를 높여줘") - 이런 입력은 needs_clarification()이 짧은 검색어 휴리스틱에
    걸려 True가 되므로, 이 가드가 없으면 search.danawa.com의 10초 Crawl-delay와
    DeepSeek facet 추출까지 그대로 타 버린다(프론트가 /decide/stream보다 먼저
    이 엔드포인트를 호출하므로 실제 체감 지연의 대부분이 여기서 생겼다).

    base_query(2026-08-13, 속도 개선) - 여러 라운드에 걸쳐 좁혀나갈 때(예: "핸드폰"
    -> "핸드폰 삼성전자" -> ...) 프론트가 그 드릴다운의 맨 처음 검색어를 실어 보낸다.
    query 대신 base_query로 검색하면 search.danawa.com의 1시간 캐시(fetchers.
    danawa_search._cache)가 맞을 확률이 높아 10초 Crawl-delay를 건너뛰고, 그 결과를
    query에서 base_query에 없는 단어들로 로컬 필터링해 재사용한다 - 실제 최종
    가격 조회(run_danawa_only_debate*)는 이 캐시/필터링을 안 쓰고 항상 정확한
    검색을 새로 한다.

    정적 facet 캐시(2026-08-16, 속도 개선) - "아이폰"처럼 자주 검색될 유명
    카테고리는 facet_cache.lookup()이 정규식 매칭만으로 즉시 답한다(검색도
    DeepSeek 호출도 없음) - 실제 검색+추출 경로(아래)는 이 캐시에 없는 질의만
    타게 된다. 매치 안 되면 None이라 기존 동작 그대로 이어진다."""
    if not needs_clarification(query) or is_non_product_chitchat(query):
        return ClarifyResponse(query=query, options=ClarifyOptions())

    static_facets = facet_cache.lookup(query)
    if static_facets is not None:
        static_facets = _strip_query_answered_options(query, static_facets)
        return ClarifyResponse(query=query, options=ClarifyOptions(facets=static_facets))

    search_query = base_query if base_query and base_query.strip() else query
    items = await price_table_module._search_danawa_items(
        search_query, limit=price_table_module.CLARIFY_SEARCH_LIMIT
    )
    # items가 비었으면(검색 실패/차단) 카테고리 집계도 같은 이유로 비어있을
    # 것이므로 굳이 다시 부르지 않는다 - 실패한 요청은 캐시되지 않아서
    # (danawa_search._fetch_entry) 여기서 또 부르면 이미 실패한 요청을
    # 그대로 재시도해 불필요한 지연만 늘린다.
    categories = await price_table_module._search_danawa_categories(search_query) if items else []

    effective_category = _select_effective_category_name(query, categories)
    if effective_category:
        # 카테고리를 이미 골랐으면(질의 텍스트에 그 이름이 있으면) 브랜드
        # 전체 표본이 아니라 그 카테고리로 좁힌 실제 표본으로 나머지
        # facet(모델/용량 등)을 뽑는다 - 안 그러면 브랜드 전체 표본에는
        # 그 카테고리 상품이 아예 없을 수 있어(예: "샤오미" 상위 40개엔
        # 휴대폰이 하나도 없다) "모델" 같은 축이 전혀 다른 카테고리 상품으로
        # 채워진다(2026-08-18 사용자 리포트: "공기청정기 · 미 패드5처럼
        # 존재하지 않는 상품으로 매핑돼"). 이렇게 표본 자체를 카테고리로
        # 좁혀두면, 그 안에서 다시 계산되는 _attach_facet_crossfilter가
        # "모델을 고르면 용량도 그에 맞게 좁혀지는" 것까지 자연히 따라온다 -
        # 표본이 이미 그 카테고리 상품뿐이라 별도 로직이 필요 없다.
        # search.danawa.com Crawl-delay(10초)를 이 라운드에 한 번 더
        # 감수한다 - _ecosystem_name_pool과 같은 트레이드오프.
        category_items = await price_table_module._search_danawa_items(
            f"{search_query} {effective_category}", limit=price_table_module.CLARIFY_SEARCH_LIMIT
        )
        if len(category_items) >= MIN_FILTERED_CLARIFY_ITEMS:
            items = category_items

    if base_query and base_query.strip() and base_query.strip() != query.strip():
        items = _filter_items_by_extra_terms(items, query, base_query)
    names = [item["product_name"] for item in items]
    facets = await _extract_facets(query, names, persona)
    facets = _strip_query_answered_options(query, facets)
    # _apply_category_breakdown은 _strip_query_answered_options 뒤에 와야 한다 -
    # 실측 카테고리 facet은 스스로 "이미 고른 값"을 정확히(부모 대분류 이름과의
    # 우연한 부분 문자열 겹침을 걸러내고) 판정해 만들어지는데(_category_breakdown_facet
    # 참고), 앞서 오면 이 일반 stripping이 그 정확한 결과를 다시 (덜 정확하게)
    # 건드려버린다.
    facets = _apply_category_breakdown(facets, query, categories)
    return ClarifyResponse(query=query, options=ClarifyOptions(facets=facets))


def _select_category_group(
    query: str, groups: list[danawa_search.DanawaCategoryGroup]
) -> danawa_search.DanawaCategoryGroup | None:
    """이전 라운드에서 사용자가 카테고리 하나를 이미 골라 질의 텍스트에 그
    이름이 그대로 들어있으면 그 카테고리를 돌려준다(_facet_resolved와 같은
    텍스트 포함 판정). 여러 개가 동시에 부분 문자열로 걸리면(예: "태블릿/
    휴대폰"을 고른 뒤에도 "태블릿"이라는 별개 중분류 이름이 그 안에 우연히
    포함돼 같이 매치됨) 가장 긴(=가장 구체적인) 이름을 우선한다."""
    matched = [g for g in groups if g["name"].casefold() in query.casefold()]
    if not matched:
        return None
    return max(matched, key=lambda g: len(g["name"]))


def _select_effective_category_name(
    query: str, categories: list[danawa_search.DanawaCategoryGroup]
) -> str | None:
    """이미 고른 카테고리 중 가장 구체적인(중분류 > 대분류) 이름을 돌려준다 -
    대분류/중분류 둘 다 같은 다나와 응답에 이미 들어있어(parse_category_breakdown)
    추가 조회 없이 판단 가능하다. 이 이름을 브랜드 질의에 붙여 다나와를 다시
    검색하면(check_clarify_facets) "모델"/"용량" 등 나머지 facet이 그
    카테고리에 실제로 속하는 상품만으로 뽑힌다."""
    top = _select_category_group(query, categories)
    if top is None:
        return None
    # top["name"] 자체가 우연히 자기 중분류 이름을 부분 문자열로 포함할 수
    # 있다(예: 대분류 "태블릿/휴대폰"은 중분류 "휴대폰"을 이미 포함한다) - 그
    # 부분을 지우고 나머지에서만 중분류를 찾아야, 대분류만 고른 상태(아직
    # 중분류는 안 고름)를 중분류까지 고른 것으로 착각하지 않는다.
    remainder = query.casefold().replace(top["name"].casefold(), "", 1)
    sub = _select_category_group(remainder, top["subcategories"])
    return sub["name"] if sub is not None else top["name"]


def _category_breakdown_facet(
    query: str, categories: list[danawa_search.DanawaCategoryGroup]
) -> ClarifyFacet | None:
    """다나와 검색결과의 실측 카테고리 집계로 "카테고리" facet을 만든다
    (2026-08-18 사용자 리포트: "샤오미"를 검색하면 AI 상세검색 카테고리에
    휴대폰이 안 나온다 - DeepSeek이 보는 상품명 표본(_extract_facets, 최대
    90개)이 다나와 검색 순위 상위권에 쏠려서, 특정 대분류 상품이 표본에 아예
    안 걸리면 그 카테고리는 영원히 못 본다). 이 집계는 표본이 아니라 다나와
    자체 카테고리 색인이라 표본 편향에서 자유롭다. 대분류가 2개 미만이면
    (=애초에 여러 카테고리로 안 갈린다는 뜻) None."""
    selected = _select_category_group(query, categories)
    if selected is None:
        groups = categories
    else:
        # selected["name"](대분류) 자체가 우연히 자기 중분류 이름을 부분
        # 문자열로 포함할 수 있다(예: 대분류 "태블릿/휴대폰"은 중분류
        # "휴대폰"을 이미 포함한다) - 그 부분을 지운 나머지에서만 "이미 고른
        # 중분류"를 판정해야, 대분류만 고른 시점(아직 중분류는 안 고름)에
        # 그 안에 우연히 포함된 중분류 옵션이 "이미 답함"으로 잘못 사라지지
        # 않는다(_select_effective_category_name과 같은 이유).
        remainder = query.casefold().replace(selected["name"].casefold(), "", 1)
        groups = [g for g in selected["subcategories"] if g["name"].casefold() not in remainder]
    groups = [g for g in groups if g["count"] > 0]
    if len(groups) < 2:
        return None
    groups = sorted(groups, key=lambda g: g["count"], reverse=True)
    return ClarifyFacet(label="카테고리", options=[g["name"] for g in groups])


def _apply_category_breakdown(
    facets: list[ClarifyFacet], query: str, categories: list[danawa_search.DanawaCategoryGroup]
) -> list[ClarifyFacet]:
    """DeepSeek이 뽑은 facets에 실측 카테고리 facet을 끼워 넣는다 - DeepSeek이
    이미 "카테고리"(또는 동의어) 라벨로 뭔가 뽑았으면 그 옵션을 실측값으로
    교체하고, 없으면 맨 앞에 새로 추가한다(카테고리는 가장 상위 질문이라
    다른 축보다 먼저 물어보는 게 자연스럽다)."""
    injected = _category_breakdown_facet(query, categories)
    if injected is None:
        return facets
    result = [injected if f.label == injected.label else f for f in facets]
    if not any(f.label == injected.label for f in facets):
        result.insert(0, injected)
    return result


def _facet_options_for_query(query: str, facet: ClarifyFacet) -> list[str]:
    """options_by_selection(_attach_facet_crossfilter가 이미 계산해둔, 다른
    facet 값을 고르면 이 facet이 어떻게 좁혀지는지)의 셀렉터 키가 질의
    텍스트에 이미 그대로 들어있으면 - 그 축은 사용자가 이미 답한 것이므로 -
    그 선택에 대응하는 좁혀진 옵션만 남긴다(2026-08-16, 그라운딩 회귀 파일럿
    50개 중 발견: "햇반 백미 210g 24개"처럼 용량·수량을 이미 구체적으로
    적었는데도 브랜드 facet의 원본 옵션(CJ제일제당/시아스/하림)이 안 좁혀진
    채 그대로 남아 불필요하게 되물었다 - crossfilter 데이터 자체는
    "210g 24개" 선택 시 CJ제일제당 하나로 좁혀진다는 걸 이미 알고 있었는데
    _facet_resolved/_is_ambiguous_facets가 그 데이터를 안 쓰고 있었다).
    매치되는 셀렉터가 여럿이면(서로 다른 축이 각각 질의에 있으면) 교집합을
    쓴다. 매치가 하나도 없거나 교집합이 비면(모순되는 신호) 원본 그대로
    돌려준다 - 잘못 좁혀서 정말 필요한 되묻기를 건너뛰는 것보다, 안전하게
    그대로 두는 쪽이 낫다."""
    by_selection = facet.options_by_selection or {}
    matched = [set(options) for selector, options in by_selection.items() if selector.casefold() in query.casefold()]
    if not matched:
        return facet.options
    narrowed = set.intersection(*matched)
    return [o for o in facet.options if o in narrowed] or facet.options


def _facet_resolved(query: str, facet: ClarifyFacet) -> bool:
    """이 facet의 옵션 중 하나라도 이미 질의 텍스트에 그대로 들어있으면
    (=사용자가 이미 답한 값이면) True - _strip_resolved_options가 브랜드/제품에
    쓰던 텍스트 매칭 판정을 라벨이 고정되지 않은 facet에 대해 일반화한 것.
    (2026-08-16 확장) 또는 다른 축의 선택으로 crossfilter가 이 facet을 옵션
    1개 이하로 좁혔으면(=사실상 답이 하나로 정해졌으면)도 True."""
    if any(o.casefold() in query.casefold() for o in facet.options):
        return True
    return len(_facet_options_for_query(query, facet)) <= 1


def _resolved_facet_count(query: str, facets: list[ClarifyFacet]) -> int:
    """_resolved_dimension_count의 facet 버전 - 이번 라운드에 새로 뽑힌 facet
    중 이미 질의 텍스트에 반영된(=사용자가 이미 답한) 게 몇 개인지 센다.
    스트리핑 전에 호출해야 한다(스트리핑 후엔 이미 다 비워져 있어 셀 수 없음)."""
    return sum(1 for f in facets if _facet_resolved(query, f))


def _strip_resolved_facets(query: str, facets: list[ClarifyFacet]) -> list[ClarifyFacet]:
    """_strip_resolved_options의 facet 버전 - 이미 질의에 반영된 facet은
    선택지 목록에서 아예 뺀다(같은 질문을 다시 보여주지 않기 위한 보조
    방어선 - _strip_resolved_options 참고)."""
    return [f for f in facets if not _facet_resolved(query, f)]


def _is_ambiguous_facets(query: str, facets: list[ClarifyFacet]) -> bool:
    """_is_ambiguous(adk_pipeline.py)의 facet 버전 - facet 중 하나라도 옵션이
    2개 이상이고 아직 질의에 반영되지 않았으면 사용자에게 물어볼 만큼
    애매하다고 본다."""
    return any(len(f.options) > 1 and not _facet_resolved(query, f) for f in facets)


def _filter_listing_pages(results: list[SearchResult]) -> list[SearchResult]:
    """검색결과 목록/사이트내검색 페이지(예: search.11st.co.kr/...?kwd=)를 clarify
    추출 대상에서 뺀다. 이런 페이지는 사이드바에 "함께 본 상품"류의 전혀 무관한
    상품이 잔뜩 섞여 있어서, GPT가 그 안 상품명을 죄다 브랜드/제품 옵션으로
    뽑아버리는 문제가 있었다(propose 단계는 filter_candidates에서 이미
    is_generic_listing_url로 걸러왔는데, clarify 추출은 이 필터를 거치지 않고
    있었다). 전부 걸러져도 빈 리스트를 그대로 반환 — 호출부가 "옵션 없음"으로
    안전하게 처리한다."""
    return [r for r in results if not is_generic_listing_url(r.url)]


async def _extract_clarify_options(query: str, results: list[SearchResult]) -> ClarifyResponse | None:
    """검색 결과 제목에서 facet(임의 기준)을 뽑아본다. 아무것도 못 찾으면 None.
    이미 가져온 검색 결과를 그대로 받아 재검색하지 않는다 — run_single_debate가
    전체 실패했을 때 같은 결과로 이 함수를 다시 시도하는 용도로도 쓰인다.

    2026-08-16부터 check_clarify_facets(다나와 직접 검색 기반)와 같은 facet
    추출 파이프라인(_extract_facets)을 쓴다(README "장기 과제" 후속) - 이전엔
    고정 4축(브랜드/제품/용량/개수)을 Qwen으로 따로 뽑았는데, 프론트가 이미
    facet 렌더링을 완전히 지원해서(교차 필터링·검색창·페르소나 배지) 별도
    UI 변경 없이 바로 통합할 수 있었다. 입력은 여기 있는 Tavily 검색 결과
    제목을 그대로 쓴다 - check_clarify_facets처럼 다나와를 새로 검색하지
    않는다(이미 검색 결과를 들고 있어서 네트워크 호출을 하나 더 늘릴 이유가
    없다).

    카테고리 기반 축 관련성 필터링은 이번 통합에서 안 옮겼다 - facet 추출
    프롬프트(build_facet_clarify_prompt) 자체가 이미 검색 결과에 실제로 의미
    있는 축만 뽑도록 유도해서, 무의미한 facet이 거의 안 생길 가능성이 높다.
    실제로 문제가 확인되면 라벨 패턴 매칭으로 후속 추가한다(고정 4축 전용이던
    예전 구현은 죽은 코드 정리 과정에서 제거했다).

    토큰 절약(2026-08-19) - check_clarify_facets(다나와 직접 검색 경로)는
    needs_clarification()/chitchat/facet_cache 가드를 이미 거치는데, 이 함수는
    같은 _extract_facets(브랜드별 최대 15개 병렬 DeepSeek 호출 + 기종 생태계별
    2개 추가 호출까지 갈 수 있는 무거운 파이프라인)를 이 가드 없이 매 요청마다
    무조건 태우고 있었다(실측: run_stream의 검색 직후 안전망이 skip_clarify와
    무관하게 이 함수를 항상 호출 - skip_clarify=True인 후속 라운드에서도 결과가
    버려질 뿐 호출 자체는 그대로 나갔다). 같은 가드를 여기도 적용해 애초에 좁혀야
    할 필요가 없는(또는 잡담인) 질의는 DeepSeek를 한 번도 부르지 않고 즉시 끝내고,
    흔한 카테고리는 facet_cache 정적 매칭으로 대체한다."""
    if not needs_clarification(query) or is_non_product_chitchat(query):
        return None
    static_facets = facet_cache.lookup(query)
    if static_facets is not None:
        facets = _strip_query_answered_options(query, static_facets)
    else:
        filtered_results = _filter_listing_pages(results)
        names = [r.title for r in filtered_results]
        facets = await _extract_facets(query, names)
    if _resolved_facet_count(query, facets) >= _MAX_CLARIFY_ROUNDS:
        return None
    facets = _strip_resolved_facets(query, facets)
    if not facets:
        return None
    return ClarifyResponse(query=query, options=ClarifyOptions(facets=facets))


async def run_single_debate(
    query: str, skip_clarify: bool = False
) -> DecideResponse | ClarifyResponse:
    """정제→검색→제안(GPT·Gemini·DeepSeek 병렬)→필터링+병합→검증→매칭→심사
    역할 분리 파이프라인 — 실제 오케스트레이션은 adk_pipeline(ADK SequentialAgent)이
    담당한다. 이 함수는 main.py/run_debate가 기대하는 기존 시그니처를 유지하는
    얇은 래퍼.

    skip_clarify(2026-08 통합 병합) - adk_pipeline.run() 참고. 이미 한 라운드
    이상 답한 후속 턴에서 내부 애매함 판정이 다시 clarify를 띄우는 걸 막는다."""
    return await adk_pipeline.run(query, skip_clarify=skip_clarify)


async def run_single_debate_stream(
    query: str, skip_clarify: bool = False
) -> AsyncIterator[dict[str, Any]]:
    """run_single_debate와 같은 결과를 만들지만, 파이프라인 단계마다(정제/검색/
    제안/검증/심사) NDJSON 이벤트를 흘려보낸다 — adk_pipeline.run_stream이 실제
    오케스트레이션과 이벤트 번역을 담당."""
    async for event in adk_pipeline.run_stream(query, skip_clarify=skip_clarify):
        yield event


async def _danawa_results_as_search_results(query: str, limit: int) -> list[SearchResult]:
    """Tavily(search_module.search, danawa.com 도메인 한정)가 이 질의를 못 찾을
    때 쓰는 대체 검색 소스(사용자 리포트, 2026-08-18: "다나와에 검색하면
    나오는데 뭐가 문제인거야?" - 실제로 다나와 자체 검색엔 있는 상품이 Tavily
    색인엔 없어서 대량구매 경로가 제안을 하나도 못 만든 사례가 있었다).
    check_clarify_facets 등 다른 경로가 이미 쓰는 다나와 직접 검색을 재사용해,
    같은 원리로 후보 검색 결과를 채운다."""
    try:
        items = await price_table_module._search_danawa_items(query, limit=limit)
    except Exception:
        return []
    return [
        SearchResult(
            title=item["product_name"],
            url=f"https://prod.danawa.com/info/?pcode={item['pcode']}",
            snippet=item["product_name"],
            score=None,
        )
        for item in items
    ]


async def run_bulk_debate(query: str) -> BulkDecideResponse | DecideResponse | ClarifyResponse:
    try:
        results = await search_module.search(query, max_results=10)
    except Exception:
        results = []

    if not results:
        results = await _danawa_results_as_search_results(query, price_table_module.MAX_DANAWA_URLS)

    proposals: list[BulkProposal] = list(
        await asyncio.gather(
            gpt.propose_bulk(query, results),
            groq.propose_bulk(query, results),
            deepseek.propose_bulk(query, results),
        )
    )

    try:
        decision = await judge.organize_options(query, proposals)
    except RuntimeError:
        # 대량구매로 분류됐지만(숫자+단위 휴리스틱, is_bulk_query) 제안을
        # 하나도 못 찾은 경우 - 검색 자체가 안 됐거나(위 다나와 폴백도 실패)
        # is_bulk_query()가 오판했을 수 있다(사용자 리포트, 2026-08-18: OCR로
        # 읽은 지저분한 텍스트가 숫자+단위를 포함해 대량구매로 잘못 분류됨).
        # 그대로 raw 예외를 흘려보내지 말고 단일상품 파이프라인으로 한 번 더
        # 시도한다 - 검색 결과 자체가 없으면 거기서도 결국 실패하지만, 최소한
        # NO_CANDIDATE_ERROR 같은 정돈된 한글 메시지로 끝난다.
        return await run_single_debate(query)

    if len(decision.options) <= 1:
        # is_bulk_query()는 "숫자+단위가 있으면 대량구매성"이라는 근사치
        # 휴리스틱이라, "메로나 빙그레 70mL"처럼 애초에 특정 상품 하나를
        # 가리키는 질의도 걸릴 수 있다(코드 주석에 이미 명시된 한계). 실제로
        # 갈리는 브랜드가 0~1개뿐이면 "여러 브랜드 비교"라는 전제 자체가
        # 성립하지 않으므로, 브랜드별 비교 포맷 대신 단일상품 파이프라인으로
        # 다시 판단한다.
        return await run_single_debate(query)

    price_range = _compute_price_range(decision.options)

    return BulkDecideResponse(
        query=query, proposals=proposals, decision=decision, price_range=price_range
    )


async def run_clarify(query: str) -> DecideResponse | ClarifyResponse:
    try:
        results = await search_module.search(query, max_results=10)
    except Exception:
        results = []

    clarify = await _extract_clarify_options(query, results)
    if clarify is not None:
        return clarify

    if not results:
        # 다나와 검색 자체가 이 질의에 대해 아무것도 못 찾았다 - run_single_debate가
        # 정제해서 다시 검색해도 같은 검색 엔진, 거의 같은 검색어라 또 아무것도
        # 못 찾을 가능성이 매우 높다(사용자 요청, 2026-08-15: "다른 쓸데없는 말
        # 하니까 왜이리 오래걸려" - is_non_product_chitchat이 못 잡는, 상품과
        # 무관한 임의의 텍스트까지 이 증거 기반 신호로 일반적으로 빠르게 실패
        # 처리한다). 검색+제안 3개+검증+심사 전체를 다시 태우는 대신 바로
        # 정직하게 포기한다.
        raise RuntimeError(NO_CANDIDATE_ERROR)

    return await run_single_debate(query)


async def run_brand_price(query: str, brand: str) -> BrandPriceResponse:
    try:
        results = await search_module.search(f"{query} {brand}", max_results=10)
    except Exception:
        results = []

    option = await gpt.find_lowest_price(query, brand, results)

    if option is None:
        return BrandPriceResponse(
            query=query, brand=brand, error=f"'{brand}' 브랜드 상품을 찾지 못했습니다."
        )

    return BrandPriceResponse(query=query, brand=brand, option=option)
