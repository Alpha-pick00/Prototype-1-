import logging
import re

import httpx

from . import search_cache
from .agents.base import is_generic_listing_url
from .config import settings
from .schemas import SearchResult

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# 토큰 절약(2026-08-19 실측) - 다나와 상품 페이지의 raw_content를 그대로 앞에서
# 1500자만 잘라 쓰면, 실제로는 그 1500자가 거의 전부 로그인/카테고리 메뉴/공유
# 버튼/저작권 문구 같은 전 페이지 공통 boilerplate였다("나이키 에어포스1" 실측:
# 1500자 중 실제 상품 정보는 "상세 스펙: 운동화 / 여성용(W) / ..." 한 줄뿐).
# 이 문구는 상품마다 안 바뀌므로 후보 여러 개를 한 프롬프트에 넣으면 같은 텍스트가
# 반복돼 토큰만 더 낭비된다. 알려진 고정 문구를 줄 단위로 걸러내고, 남은 텍스트를
# 자른다 - 상품명·카테고리 트리처럼 페이지마다 바뀌는 텍스트는 안 건드린다(안전한
# 쪽으로: 걸러도 되는지 확신 없는 줄은 그대로 남긴다).
_DANAWA_BOILERPLATE_LINES = frozenset({
    "메인 메뉴로 바로가기 본문으로 바로가기",
    "에누리", "몰테일", "메이크샵",
    "다나와 가격비교 CI", "최근",
    "로그인", "회원가입", "마이페이지", "쪽지", "광고센터", "고객센터",
    "다나와 앱", "다나와 앱 서비스 목록", "다나와 앱 서비스 목록 닫기",
    "가격비교 장터 PC견적 자동차", "다나와 APP",
    "다나와 장터", "PC견적", "자동차", "QR코드", "빈 이미지",
    "PC구매상담", "쇼핑기획전", "커뮤니티", "이벤트 / 체험단",
    "서비스더보기", "서비스 전체보기",
    "+ 샵다나와", "+ 브랜드로그", "+ 중고마켓", "+ 동영상", "+ 중고매입",
    "+ 모바일 앱", "+ 다나와AS", "+ PC26", "+ 장터",
    "전체 카테고리", "홈",
    "컨텐츠 상단으로 이동   컨텐츠 하단으로 이동", "로딩중",
    "관심", "공유", "공유하기", "레이어 닫기",
    "+ 카카오톡", "+ 라인", "+ 페이스북", "+ X", "+ 밴드", "복사",
    "URL이 복사되었습니다.", "원하는 곳에 붙여넣기(Ctrl+V)하세요.",
    "신고", "인쇄", "동영상 재생",
    "최대12개월 무이자할부",
    "확인 할 수 있어요.",
    "상품 상세정보", "대체 텍스트 노출",
    "광고", "가격비교", "쇼핑몰 선택", "쇼핑몰 정보",
    "의견/리뷰", "소모품/액세서리", "뉴스/커뮤니티", "연관상품",
})

_DANAWA_BOILERPLATE_SUBSTRINGS = (
    "다나와 가격비교 No.1 가격비교사이트",
    "다나와 장터 언제 어디서나",
    "팔거나 살 수 있는 스마트한 모바일 장터",
    "다나와 PC견적 PC조립을 위한",
    "실시간 최저가로 손쉽게 조립PC",
    "다나와 자동차 대한민국 최대 규모",
    "견적평가, 중고차 매물 검색",
    "자동차 관련 소식을 받아보실 수 있습니다",
    "인터넷 요금 비교, 이제 다나와에서 시작하세요",
    "카드결제, 쿠팡 와우회원",
    "결제 금액에 따라 무이자 혜택",
    "다나와 가격비교 앱 > 알림에서",
    "이미지출처",
    "우리 집 조건에 맞는 인터넷 요금",
    "콘텐츠산업 진흥법",
    "전자우편 수집 프로그램",
    "정보통신망법에 의해 형사처벌",
)

# 상품명_이미지 / 상품명_동영상_이미지 형태의 이미지 alt 텍스트 - 상품명이 매번
# 달라 리터럴 목록에 못 넣으므로 접미사 패턴으로 잡는다.
_IMAGE_ALT_SUFFIX_RE = re.compile(r"_(이미지|동영상_이미지)\s*$")


def _strip_danawa_boilerplate(text: str) -> str:
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _DANAWA_BOILERPLATE_LINES:
            continue
        if any(s in stripped for s in _DANAWA_BOILERPLATE_SUBSTRINGS):
            continue
        if _IMAGE_ALT_SUFFIX_RE.search(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)

# frontend/src/app/components/About.tsx의 "We Compare across" 목록과 동일.
# 15개 쇼핑몰 각각의 서로 다른 페이지 구조를 스니펫만 보고 파싱하다 보니 엉뚱한
# 상품/가격이 섞이는 문제가 있었다 — 다나와는 그 자체로 여러 판매처의 가격을
# 한 페이지에서 비교해주는 가격비교 사이트라, 도메인을 다나와 하나로 좁혀서
# 결과의 일관성과 정확도를 우선한다.
# enuri.com은 한때 이중화 목적으로 추가했었으나(가격비교 사이트, 다나와 차단
# 대비) 어댑터 없이 검색 결과 노출 정도로만 쓰였고, 실제로는 비교 대상에서
# 제외하기로 해 뺐다(2026-08-15).
RETAILER_DOMAINS = [
    "danawa.com",
]

# 상품 상세/가격 정보가 없는 콘텐츠 도메인. include_domains에 danawa.com처럼 상위
# 도메인을 넣으면 이 서브도메인들이 검색 순위를 독점해 실제 쇼핑몰 상품 페이지를
# 밀어내는 현상이 있어 명시적으로 제외한다.
EXCLUDE_DOMAINS = [
    "dpg.danawa.com",  # 다나와 매거진/리뷰 블로그, 가격 정보 없음
    "search.danawa.com",  # 검색결과 목록 페이지 (is_generic_listing_url로도 걸리지만 애초에 제외)
    # 다나와 "쇼핑기획전" 프로모션 페이지 - 특정 상품이 아니라 카테고리 전체를
    # 홍보하는 콘텐츠 페이지다(2026-08-18, 그라운딩 회귀 파일럿에서 발견: "삼성전자
    # 비스포크 냉장고 4도어" 검색 결과 12건 중 7건이 이 도메인이었고, 실제 상품
    # 가격비교 페이지는 액세서리 1건뿐이었다 - propose가 진짜 후보를 고를 수 없었다).
    "plan.danawa.com",
]


async def _tavily_search(
    query: str, max_results: int, domains: list[str] = RETAILER_DOMAINS
) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TAVILY_URL,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_domains": domains,
                "exclude_domains": EXCLUDE_DOMAINS,
                "include_raw_content": "text",
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for r in data.get("results", []):
        # 카테고리 목록 페이지(예: prod.danawa.com/list?cate=)는 특정 상품 하나를
        # 가리키지 않아 가격/판매처 정보가 없다 - propose 단계는 지금까지 이걸
        # 후보로 받아들인 뒤(is_generic_listing_url, agents/base.py)에야 걸러냈는데,
        # "이어폰"처럼 넓은 카테고리어는 Tavily 결과 자체가 이런 목록 페이지로
        # 뒤덮여 있어(실측 2026-08-19: "이어폰" 검색 8건 전부가 목록 페이지) propose가
        # 볼 수 있는 실제 상품이 하나도 없는 채로 프롬프트가 채워졌다 - 3개 모델이
        # 전부 후보를 못 만들어 "적절한 상품 후보를 찾지 못했습니다"로 끝났다.
        # 검색 결과를 프롬프트에 넣기 전에 걸러야, propose가 애초에 실제 상품만 본다.
        if is_generic_listing_url(r["url"]):
            continue
        raw = r.get("raw_content") or ""
        snippet = r.get("content", "")
        # raw_content가 있으면 스니펫보다 정보가 많으므로 우선 사용 - boilerplate를
        # 먼저 걷어낸 뒤에 자른다(순서 중요: 자른 뒤에 걷어내면 이미 잘려나간
        # 뒷부분의 실제 상품 정보를 영영 못 건짐).
        text = _strip_danawa_boilerplate(raw)[:1500] if raw else snippet
        results.append(SearchResult(title=r["title"], url=r["url"], snippet=text, score=r.get("score")))
    return results


async def _fetch(query: str) -> list[SearchResult]:
    """Tavily를 호출해 검색 결과를 가져온다. 캐시를 거치지 않는 순수 조회 —
    search()의 캐시 미스 경로와 refresh()의 강제 갱신 경로가 이 함수를 공유한다."""
    return await _tavily_search(query, search_cache.FETCH_SIZE)


async def search(query: str, max_results: int = 12) -> list[SearchResult]:
    """같은 질의가 반복되면 search_cache에서 재사용한다 — 항상 FETCH_SIZE만큼
    받아서 캐시해두고, 더 적은 max_results를 요청한 호출은 앞에서 잘라 쓴다."""
    cached = search_cache.get(query)
    if cached is not None:
        return cached[:max_results]

    merged = await _fetch(query)
    search_cache.set(query, merged)
    return merged[:max_results]


async def refresh(query: str) -> None:
    """TTL을 기다리지 않고 캐시를 강제로 새로 채운다 — 인기 질의 우선 갱신
    스케줄러(popularity_scheduler)가 사용."""
    merged = await _fetch(query)
    search_cache.set(query, merged)


async def extract(url: str) -> str | None:
    """URL 하나의 전체 페이지 본문을 가져온다. 후보를 하나로 좁힌 뒤 가격을 재확인할 때 사용."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            TAVILY_EXTRACT_URL,
            json={"api_key": settings.tavily_api_key, "urls": [url]},
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if not results:
        return None
    raw = results[0].get("raw_content")
    return _strip_danawa_boilerplate(raw) if raw else raw


COUPANG_DOMAINS = ["coupang.com"]
_COUPANG_MAX_RESULTS = 5


async def search_coupang(query: str) -> list[SearchResult]:
    """challenge 단계 그라운딩 보조 신호(2026-08-16, "그라운딩 성능을 높여줘") -
    다나와 검색과 완전히 별도로 쿠팡에 한정해 Tavily를 직접 호출한다. search()의
    캐시/임베딩 유사도 매칭은 안 쓴다 - 참고 신호일 뿐이라 캐시 재사용 이점이
    크지 않고, 매 요청 최신 재고를 보는 게 더 정확하다. 쿠팡 페이지를 직접
    파싱해 가격/후보를 뽑지는 않는다 - Tavily 스니펫만 challenge LLM에게
    참고 자료로 넘긴다(다나와 하나로 리테일러 도메인을 좁힌 이유였던 "페이지
    구조가 달라 스니펫만으로 파싱하면 엉뚱한 상품/가격이 섞이는 문제"를
    재현하지 않기 위함). 실패해도 조용히 빈 리스트 - challenge는 이 신호 없이도
    기존 방식대로 동작한다."""
    try:
        return await _tavily_search(query, _COUPANG_MAX_RESULTS, domains=COUPANG_DOMAINS)
    except Exception:
        logger.warning("쿠팡 교차 확인 검색 실패: %r", query, exc_info=True)
        return []


_UNRESTRICTED_MAX_RESULTS = 3


async def search_unrestricted(query: str) -> list[SearchResult]:
    """다나와 한정 검색(search())이 아무것도 못 찾았을 때 쓰는 최후 폴백
    (사용자 요청, 2026-08-19: "검색 알고리즘으로 적절한 상품을 찾을 수 없는
    경우에는 구글 쇼핑에서 사용자 쿼리를 따로 검색해서... 다나와에서
    가져오게") - 구글 쇼핑 전용 API는 무료 티어가 없고(SerpAPI 등 유료
    서드파티), 직접 스크래핑은 danawa.com에서 이미 겪은 IP 차단 위험을 또
    다른 도메인에 반복하는 셈이라(2026-08-18 실측: AWS IP가 데이터센터
    대역이라 차단당함), 이미 쓰고 있는 Tavily를 도메인 제한 없이(전체 웹)
    호출해 같은 목적(질의에 맞는 실제 상품/브랜드명 발견)을 달성한다.

    이 결과 자체를 후보로 쓰지 않는다 - 다나와 URL이 아니면 구매 링크를
    만들 수 없다(파이프라인 전체가 다나와 bridge_url 해석에 의존). 호출부
    (adk_pipeline._broad_web_fallback_search)가 여기서 발견한 상품명을
    다나와에 다시 검색해 실측 후보로 바꾼다. search()와 달리 캐시를 쓰지
    않는다 - 이미 드문 최후 폴백이라 캐시 재사용 이점이 크지 않다. 실패해도
    조용히 빈 리스트 - 이 경로가 없어도 기존 실패 처리(NO_CANDIDATE_ERROR)
    그대로 동작한다."""
    try:
        return await _tavily_search(query, _UNRESTRICTED_MAX_RESULTS, domains=[])
    except Exception:
        logger.warning("비제한 폴백 검색 실패: %r", query, exc_info=True)
        return []


NAVER_DOMAINS = ["shopping.naver.com"]
_NAVER_MAX_RESULTS = 5


async def search_naver(query: str) -> list[SearchResult]:
    """쿠팡(search_coupang)과 동일한 패턴의 두 번째 소프트 그라운딩 신호
    (2026-08-16, "다나와 단일 실측 소스에 대한 의존도를 낮추도록") - 다나와
    실측가가 유일한 "확정" 소스이고 쿠팡 하나만으로는 교차 확인 대상이 한
    곳뿐이라, 서로 다른 두 번째 독립 쇼핑몰을 더해 challenge 판단의 참고
    자료를 넓힌다. 쿠팡과 마찬가지로 페이지를 파싱해 후보를 만들지 않고
    Tavily 스니펫만 참고용으로 넘긴다 - 15개 리테일러를 다나와로 좁혔던
    이유(스니펫만으로 파싱하면 엉뚱한 상품/가격이 섞임)를 반복하지 않기
    위함. 실패해도 조용히 빈 리스트."""
    try:
        return await _tavily_search(query, _NAVER_MAX_RESULTS, domains=NAVER_DOMAINS)
    except Exception:
        logger.warning("네이버쇼핑 교차 확인 검색 실패: %r", query, exc_info=True)
        return []
