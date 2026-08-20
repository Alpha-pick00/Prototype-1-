import asyncio
import logging
import re
from typing import Any, AsyncIterator

from fetchers import elevenst

from . import embeddings
from . import facet_cache
from . import llm_cache
from . import price_table as price_table_module
from .agents import deepseek, gpt
from .intent import is_non_product_chitchat, needs_clarification
from .schemas import (
    BrandOption,
    BrandPriceResponse,
    ClarifyFacet,
    ClarifyOptions,
    ClarifyResponse,
    Decision,
    DecideResponse,
    Proposal,
)

logger = logging.getLogger(__name__)


async def _rank_by_relevance(
    query: str, items: list[elevenst.ElevenstSearchItem]
) -> list[elevenst.ElevenstSearchItem]:
    """Qwen 임베딩 코사인 유사도로 관련도 내림차순 정렬한다 - "함께 추천"되는
    관련상품 목록(proposals)의 순서이자, 추천 Agent에게 넘기는 후보 순서다.
    임베딩이 실패하면(키 없음·API 오류) 원래 순서(11번가가 준 순서) 그대로
    돌려준다 - 정렬 실패가 검색 자체를 막으면 안 된다."""
    names = [it["product_name"] for it in items]
    vectors = await embeddings.embed([query, *names])
    if vectors is None:
        return items
    query_vec, *item_vecs = vectors
    scored = sorted(
        zip(items, item_vecs), key=lambda pair: embeddings.cosine_similarity(query_vec, pair[1]), reverse=True
    )
    return [it for it, _ in scored]


async def _search_candidates(query: str, base_query: str | None) -> list[elevenst.ElevenstSearchItem]:
    """HITL 드릴다운(2026-08-20 재설계, "쿼리 재구성해서 검색하는 거나
    다름없다" 지적) - base_query가 없으면(단발 질의) 그 질의로 좁게(10개)
    검색해 끝낸다. base_query가 있으면(AI 상세검색을 거쳐 facet을 answer로
    덧붙인 드릴다운 질의) 매번 새로 조합된 전체 문자열로 재검색하지 않는다 -
    check_clarify_facets와 똑같이 안정적인 base_query로 넓게(90개) 검색한
    뒤, 그 위에 사용자가 덧붙인 답(facet 값들)을 _filter_items_by_extra_terms로
    구조적으로(순수 로컬 필터링, 추가 네트워크 요청 없음) 좁힌다 - 매 라운드
    새 문자열을 만들어 11번가를 다시 때리는 대신, 검증된 후보군 자체를
    필터링해 나간다."""
    if base_query and base_query.strip() and base_query.strip() != query.strip():
        items = await elevenst.search_elevenst(base_query, limit=price_table_module.CLARIFY_SEARCH_LIMIT)
        return _filter_items_by_extra_terms(items, query, base_query)
    return await elevenst.search_elevenst(query, limit=10)


async def run_elevenst_only_debate(query: str, base_query: str | None = None) -> DecideResponse:
    """11번가 오픈 API(ProductSearch)로 검색한다(_search_candidates - base_query가
    있으면 재검색 대신 구조적 필터링). _product_name_matches로 질의와 실제로
    맞는 상품만 후보(검증된 후보군)로 남기고, Qwen 임베딩으로 관련도순
    정렬한 뒤(_rank_by_relevance) 그 순서 그대로 proposals에 담아 "관련 상품"
    목록으로 노출한다. 최종 추천은 추천 Agent(gpt.recommend_best, 가격뿐
    아니라 리뷰·구매만족도까지 고려)가 고르고, 실패하면(키 없음·API 오류)
    최저가 규칙 기반으로 폴백한다."""
    items = await _search_candidates(query, base_query)
    relevant = [it for it in items if price_table_module._product_name_matches(query, it["product_name"])]
    if not relevant:
        raise RuntimeError(f"11번가에서 '{query}'에 대해 관련성 있는 상품을 찾지 못했다.")

    ranked = await _rank_by_relevance(query, relevant)

    recommended = await gpt.recommend_best(query, ranked)
    if recommended is not None:
        index, llm_reasoning = recommended
        best = ranked[index]
        reasoning = (
            f"11번가 실측 검증 후보 중 추천 Agent(Qwen)가 선택 - {llm_reasoning}"
            if llm_reasoning
            else "11번가 실측 검증 후보 중 추천 Agent(Qwen)가 선택"
        )
    else:
        best = min(ranked, key=lambda it: it["price_krw"])
        reasoning = "11번가 오픈 API(ProductSearch) 실측 - 추천 Agent 응답 실패로 최저가 규칙 기반 선택"

    decision = Decision(
        product_name=best["product_name"],
        price=f"{best['price_krw']:,}원",
        retailer=best["seller"],
        url=best["url"],
        reasoning=reasoning,
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
            reasoning="11번가 오픈 API 검증 결과 (관련도순 - 함께 볼만한 상품)",
            verified=True,
        )
        for it in ranked
    ]
    return DecideResponse(query=query, proposals=proposals, decision=decision)


async def run_elevenst_only_debate_stream(
    query: str, base_query: str | None = None
) -> AsyncIterator[dict[str, Any]]:
    """run_elevenst_only_debate()의 스트리밍 버전 - 메인 검색 흐름
    (/decide/stream)이 이 경로를 쓴다. status 이벤트 하나(진행 표시용) 후
    final 이벤트 하나를 내보낸다."""
    yield {"type": "status", "stage": "searching"}
    try:
        result = await run_elevenst_only_debate(query, base_query=base_query)
    except RuntimeError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    yield {"type": "final", "result": result.model_dump()}


# check_clarify_facets()의 base_query 재사용 필터링 전용(사용자 요청, 2026-08-13:
# "조금 더 빠르게 검색기능이 되면 좋겠어"). 필터링 결과가 이보다 적으면 표본이
# 너무 좁아 facet 품질이 나빠질 수 있으니, 필터링을 포기하고 base_query의
# 넓은 표본을 그대로 쓴다(추가 검색은 하지 않는다 - 속도가 이 최적화의 목적이라
# 여기서 또 검색을 때리면 본전도 못 찾는다).
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
    items: list[elevenst.ElevenstSearchItem], query: str, base_query: str
) -> list[elevenst.ElevenstSearchItem]:
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
# 이보다 표본이 적으면 그 생태계는 11번가 보충 검색을 한 번 더 돌린다(아래
# _ecosystem_name_pool) - "핸드폰 케이스"처럼 한쪽 생태계(갤럭시)가 검색
# 결과 대부분을 차지하고 다른쪽(아이폰)은 거의 없는 경우가 실제로 있다 -
# facet 추출/균형 로직이 완벽해도 원본 표본에 그 생태계 매물 자체가 거의
# 없으면 소용없다.
_DEVICE_ECOSYSTEM_SUPPLEMENT_MIN_ITEMS = 3


def _items_for_substring(term: str, names: list[str]) -> list[str]:
    nt = _normalize_for_match(term)
    return [name for name in names if nt in _normalize_for_match(name)]


async def _ecosystem_name_pool(eco: str, names: list[str], query: str) -> list[str]:
    """이 생태계(갤럭시/아이폰) 상품명이 기존 표본에 부족하면, 그 생태계 이름을
    검색어 앞에 붙여 11번가에 보충 검색을 한 번 더 돌려 실제 매물을 채운다.
    이미 충분하면(>= _DEVICE_ECOSYSTEM_SUPPLEMENT_MIN_ITEMS) 추가 네트워크
    요청 없이 기존 표본만 쓴다. 11번가 검색 자체가 실패해도(네트워크 오류 등)
    기존 표본만으로 계속 진행한다."""
    pool = _items_for_substring(eco, names)
    if len(pool) >= _DEVICE_ECOSYSTEM_SUPPLEMENT_MIN_ITEMS:
        return pool
    try:
        items = await price_table_module._search_elevenst_items(
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


_FACET_CACHE_NAMESPACE = "clarify_facets"


async def _extract_facets(
    query: str, names: list[str], persona: dict[str, str] | None = None
) -> list[ClarifyFacet]:
    """상품명 목록(출처 무관)에서 facet을 뽑는 공유 파이프라인 -
    check_clarify_facets(11번가 직접 검색)이 쓴다. 이 안에서만 최대 1(메인) +
    _MAX_BRAND_ENRICH_FANOUT(브랜드별) + 1(기종 보강) 번의 LLM 호출이 나갈 수
    있어 - 같은 질의가 반복되면(인기 검색어) llm_cache(Supabase KV+시맨틱)로
    건너뛴다. persona는 캐시된 facets 순서에 매 호출 새로 반영하므로(사용자마다
    다름) 캐시 키에는 넣지 않는다 - 캐시는 query 텍스트에만 의존한다."""
    cached = await llm_cache.exact_get(
        _FACET_CACHE_NAMESPACE, query
    ) or await llm_cache.semantic_get(_FACET_CACHE_NAMESPACE, query)
    if cached is not None:
        facets = [ClarifyFacet(**f) for f in cached["facets"]]
    else:
        facets = await deepseek.extract_facets_from_names(query, names)
        facets = await _enrich_facets_per_brand(facets, names, query)
        facets = await _enrich_device_models_by_ecosystem(facets, names, query)
        facets = _attach_facet_crossfilter(facets, names)
        incidence = _build_facet_value_incidence(facets, names)
        facets = sorted(facets, key=lambda f: _facet_sort_key(f, incidence))
        payload = {"facets": [f.model_dump() for f in facets]}
        await llm_cache.exact_set(_FACET_CACHE_NAMESPACE, query, payload)
        await llm_cache.semantic_set(_FACET_CACHE_NAMESPACE, query, payload)
    return _apply_persona_ordering(facets, persona or {})


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
    """AI 상세검색(2026-08-12 요청) - "음료수"처럼 짧고 애매한 검색어를 11번가
    실제 검색 결과 상품명에 근거해 몇 가지 기준(facet)으로 좁혀나가도록 DeepSeek에게
    물어본다(원래 Qwen으로 붙였다가, Model Studio 계정의 과금 플랜 활성화 문제로
    이미 키가 있고 바로 되는 DeepSeek로 옮겼다). run_elevenst_only_debate()/
    run_elevenst_only_debate_stream()과는 완전히 분리된 별도 진입점이다 - 그 둘은
    "LLM 호출 0번"이 테스트로 고정된 불변식이라(test_run_elevenst_only_debate_never_calls_any_llm)
    여기서 DeepSeek를 부르는 로직을 거기 안에 섞으면 안 된다. 프론트가 이 함수를
    먼저(짧은 쿼리에 한해) 호출해보고, facets가 비어 있으면(=명확한 검색어이거나
    DeepSeek 호출 실패) 그대로 elevenst-only 빠른 경로로 진행한다.

    needs_clarification()이 False면 검색조차 하지 않고 즉시 빈 결과를 반환한다 -
    대부분의(구체적인) 검색어는 이 함수를 호출해도 11번가 오픈 API 요청도,
    DeepSeek 호출도 전혀 없이 즉시 끝난다.

    "하이"처럼 상품과 무관한 인사말/잡담도 같은 이유로 즉시 빈 결과를 반환한다
    (사용자 요청, 2026-08-15: "상품으로 인식못하는 말을 들으면 처리해야하는
    속도를 높여줘") - 이런 입력은 needs_clarification()이 짧은 검색어 휴리스틱에
    걸려 True가 되므로, 이 가드가 없으면 11번가 검색과 DeepSeek facet 추출까지
    그대로 타 버린다(프론트가 /decide/stream보다 먼저 이 엔드포인트를 호출하므로
    실제 체감 지연의 대부분이 여기서 생겼다).

    base_query(2026-08-13, 속도 개선) - 여러 라운드에 걸쳐 좁혀나갈 때(예: "핸드폰"
    -> "핸드폰 삼성전자" -> ...) 프론트가 그 드릴다운의 맨 처음 검색어를 실어 보낸다.
    query 대신 base_query로 검색하면 그 결과를 query에서 base_query에 없는
    단어들로 로컬 필터링해 재사용한다 - 실제 최종 가격 조회(run_elevenst_only_debate*)는
    이 필터링을 안 쓰고 항상 정확한 검색을 새로 한다.

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
    items = await price_table_module._search_elevenst_items(
        search_query, limit=price_table_module.CLARIFY_SEARCH_LIMIT
    )
    # items가 비었으면(검색 실패/키 미설정) 카테고리 집계도 같은 이유로
    # 비어있을 것이므로 굳이 다시 부르지 않는다.
    categories = await price_table_module._search_elevenst_categories(search_query) if items else []

    # 카테고리 축은 텍스트 재검색을 쓰지 않는다(2026-08-20 재설계, "텍스트
    # 재검색 말고 다른 방식으로") - 11번가 오픈 API는 dispCtgrNo를
    # ProductSearch에 넘겨도 서버 쪽에서 실제로 필터링해주지 않고(실측 확인 -
    # TotalCount가 무시하고 그대로 나옴), 카테고리 이름("과자/간식" 등)은
    # 다나와의 대분류/중분류와 달리 상품명 텍스트에 거의 등장하지 않아
    # _filter_items_by_extra_terms 같은 순수 로컬 필터링도 통하지 않는다.
    # 그래서 카테고리 facet은 실측 breakdown(categories, 아래
    # _apply_category_breakdown)으로 정확하게 "보여주기"만 하고, 사용자가
    # 골라도 표본을 카테고리로 좁히려 하지 않는다 - 브랜드/모델/용량처럼
    # 값이 상품명에 실제로 등장하는 다른 축들이 아래 _filter_items_by_extra_terms로
    # 표본을 구조적으로 좁혀 나간다(브랜드 -> 모델 -> 용량 순으로 자연스럽게
    # 이어지는 게 그 예 - 순서 자체는 _facet_sort_key가 매 라운드 표본
    # 기준으로 동적으로 정한다).
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
    query: str, groups: list[elevenst.ElevenstCategoryGroup]
) -> elevenst.ElevenstCategoryGroup | None:
    """이전 라운드에서 사용자가 카테고리 하나를 이미 골라 질의 텍스트에 그
    이름이 그대로 들어있으면 그 카테고리를 돌려준다(_facet_resolved와 같은
    텍스트 포함 판정). 여러 개가 동시에 부분 문자열로 걸리면(예: "태블릿/
    휴대폰"을 고른 뒤에도 "태블릿"이라는 별개 중분류 이름이 그 안에 우연히
    포함돼 같이 매치됨) 가장 긴(=가장 구체적인) 이름을 우선한다."""
    matched = [g for g in groups if g["name"].casefold() in query.casefold()]
    if not matched:
        return None
    return max(matched, key=lambda g: len(g["name"]))


def _category_breakdown_facet(
    query: str, categories: list[elevenst.ElevenstCategoryGroup]
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
    facets: list[ClarifyFacet], query: str, categories: list[elevenst.ElevenstCategoryGroup]
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
    """facet 중 하나라도 옵션이 2개 이상이고 아직 질의에 반영되지 않았으면
    사용자에게 물어볼 만큼 애매하다고 본다."""
    return any(len(f.options) > 1 and not _facet_resolved(query, f) for f in facets)


async def run_brand_price(query: str, brand: str) -> BrandPriceResponse:
    """브랜드로 좁힌 검색 - run_elevenst_only_debate()와 같은 원칙(LLM 미사용,
    11번가 오픈 API 구조화 데이터 + _product_name_matches 관련성 가드)으로
    브랜드명을 검색어에 덧붙여 찾는다."""
    items = await elevenst.search_elevenst(f"{query} {brand}", limit=10)
    relevant = [it for it in items if price_table_module._product_name_matches(query, it["product_name"])]

    if not relevant:
        return BrandPriceResponse(
            query=query, brand=brand, error=f"'{brand}' 브랜드 상품을 찾지 못했습니다."
        )

    best = min(relevant, key=lambda it: it["price_krw"])
    option = BrandOption(
        brand=brand,
        product_name=best["product_name"],
        price=f"{best['price_krw']:,}원",
        retailer=best["seller"],
        url=best["url"],
        reasoning="11번가 오픈 API(ProductSearch) 실측 - LLM 미사용, 구조화 데이터 기반 최저가 선택",
    )
    return BrandPriceResponse(query=query, brand=brand, option=option)
