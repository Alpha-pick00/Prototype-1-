"""11번가 검색 결과를 파이프라인에 연결하는 계층 - 상품명 관련성 판정
(_product_name_matches)과 app.debate.check_clarify_facets가 쓰는 11번가
검색/카테고리 조회 래퍼를 담는다."""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from fetchers import elevenst
from fusion.dedup import NAME_SIMILARITY_THRESHOLD

from .exclusive_tokens import exclusive_tokens_conflict
from .spec_match import model_or_quantity_conflict

logger = logging.getLogger(__name__)

# check_clarify_facets() 전용(사용자 요청, 2026-08-12: "브랜드가 2,3개 정도만
# 뜨는데") - 상세페이지를 추가로 페치하지 않고(가격표 실측이 아니라 facet
# 추출용 상품명만 필요) 검색 결과 상품명을 최대한 많이 확보해야 브랜드/기종별
# 다양성이 나온다.
CLARIFY_SEARCH_LIMIT = 90


async def _search_elevenst_items(query: str, limit: int) -> list[elevenst.ElevenstSearchItem]:
    """app.debate.check_clarify_facets의 상품명 표본 출처 - 막히거나 실패해도
    예외를 던지지 않고 빈 리스트를 반환한다(호출자가 폴백을 따로 두지 않아도
    되도록)."""
    try:
        return await elevenst.search_elevenst(query, limit=limit)
    except elevenst.ElevenstSearchBlocked:
        logger.warning("elevenst search blocked for query=%r", query)
        return []
    except Exception:
        logger.exception("elevenst search crashed for query=%r", query)
        return []


async def _search_elevenst_categories(query: str) -> list[elevenst.ElevenstCategoryGroup]:
    """app.debate.check_clarify_facets의 "카테고리" facet 실측 출처 - 같은
    예외 안전성 계약(막히거나 실패해도 빈 리스트)."""
    try:
        return await elevenst.search_categories(query)
    except elevenst.ElevenstSearchBlocked:
        logger.warning("elevenst category breakdown search blocked for query=%r", query)
        return []
    except Exception:
        logger.exception("elevenst category breakdown search crashed for query=%r", query)
        return []


def _product_name_matches(decision_name: str, candidate_name: str) -> bool:
    """검색 결과 상품명(candidate_name)이 실제로 질의/결정된 상품명
    (decision_name)과 같은 상품인지 3단계로 판정한다 - app.debate의
    run_elevenst_only_debate/run_brand_price가 검색 결과를 후보로 받아들이기
    전 관련성 가드로 쓴다."""
    if fuzz.token_set_ratio(decision_name, candidate_name) < NAME_SIMILARITY_THRESHOLD:
        return False
    if model_or_quantity_conflict(decision_name, candidate_name):
        return False
    # 백미/발아현미처럼 순한글 단어 하나가 결정적 차이인 경우 - token_set_ratio는
    # 공통 토큰이 많으면 이런 차이를 그냥 덮어버린다(실측 93.0점, 85 통과).
    if exclusive_tokens_conflict(decision_name, candidate_name):
        return False
    return True
