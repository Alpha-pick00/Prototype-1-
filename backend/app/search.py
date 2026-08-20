import logging
import re
from urllib.parse import urlsplit

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


def _strip_lines(
    text: str,
    boilerplate_lines: frozenset[str],
    boilerplate_substrings: tuple[str, ...] = (),
    suffix_re: re.Pattern | None = None,
) -> str:
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in boilerplate_lines:
            continue
        if any(s in stripped for s in boilerplate_substrings):
            continue
        if suffix_re and suffix_re.search(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def _strip_danawa_boilerplate(text: str) -> str:
    return _strip_lines(text, _DANAWA_BOILERPLATE_LINES, _DANAWA_BOILERPLATE_SUBSTRINGS, _IMAGE_ALT_SUFFIX_RE)


# 11번가 상품 페이지(2026-08-20 실측: "광동제약 옥수수수염차" 등) - 다나와와
# 완전히 다른 boilerplate 세트를 쓴다(판매자 지표/포인트 적립/찜하기/반품·교환
# 정책·법적고지 문단 등). 판매가/가격정보/최저가/브랜드처럼 실제 값을 가리키는
# 라벨은 일부러 안 뺐다 - 바로 다음 줄의 숫자가 무슨 값인지 알 수 없게 되는
# 부작용을 막기 위해서다(다나와 쪽과 같은 원칙: 확신 없으면 남긴다).
_ELEVENST_BOILERPLATE_LINES = frozenset({
    "본문 바로가기", "[본문 바로가기](#layBodyWrap)",
    "스토어 뱃지",
    "판매자만족", "응답률", "판매량", "판매자 정보",
    ":   판매자만족",
    ":   고객이 판매자의 서비스를 평가한 리뷰 중, 4~5점의 긍정 평가의 비율 (최근 1년 기준)",
    "24시간내 응답",
    ":   상품 Q&A 문의에 24시간 내 응답한 비율 (최근 30일 기준)",
    ":   판매자의 판매건수와 판매금액을 반영하여, 판매량을 5단계로 측정 (최근 365일 기준)",
    ":   판매량은 5단계(가장 높은 판매량) ~ 1단계(낮은 판매량)로 노출",
    "## 상품 카테고리 정보", "상품 카테고리 정보",
    "브랜드패션", "트렌드패션", "뷰티", "식품", "스포츠/레저/자동차", "출산/육아",
    "가구/인테리어", "생활/건강", "가전/디지털", "여행/굿즈/e쿠폰", "해외직구", "도서/취미/펫",
    "## 상품 요약 정보", "상품 요약 정보",
    "상품이미지에 마우스를 오버하시면 확대이미지가 제공됩니다.",
    "상품에 적용된 프로모션",
    "원산지:", ":   상세설명 참조",
    "* 찜 완료", "찜 완료", ":   **찜**이 되었습니다.", ":   찜이 되었습니다.",
    "[찜한상품 전체보기](//www.11st.co.kr/interest/AuthInterestProductAction.tmall?method=getAllInterestProductInfo)",
    "찜한상품 전체보기",
    "찜해제 완료", ":   **찜**이 취소 되었습니다.", ":   찜이 취소 되었습니다.",
    "* 공유하기", "공유하기",
    "+ [페이스북](#)", "+ 페이스북", "+ [X](#)", "+ X", "+ [카카오스토리](#)", "+ 카카오스토리",
    "최대 적립 포인트", "11pay 포인트", "* 판매자 적립", "판매자 추가 적립",
    "11pay 포인트 적립 안내",
    "* 최대 리뷰 적립", "텍스트 리뷰 작성 시", "사진 리뷰 작성 시", "동영상 리뷰 작성 시",
    "리뷰작성 적립안내",
    "## 11번가 신한카드 결제할인", "최대 적립 포인트 안내",
    "추가 혜택", "빗썸 연동 구매 혜택", "빗썸 비트코인",
    "카드할인 혜택 배너",
    "### 셀러 상품", "셀러 상품",
    "판매자 인기 상품",
    "## 상세 정보", "상세 정보", "### 상품정보", "상품정보",
    "상품 일반 정보 테이블",
    "판매자가 **현금결제를 요구하면 거부**하시고 즉시 [11번가로 신고](https://cs.11st.co.kr/page/customer/faq/contents/734)해 주세요.",
    "판매자가 현금결제를 요구하면 거부하시고 즉시 11번가로 신고해 주세요.",
    "### 판매자정보(반품/교환)", "판매자정보(반품/교환)",
    "#### 반품/교환 정보", "반품/교환 정보", "반품/교환 정보 테이블",
    "#### 반품/교환 기준", "반품/교환 기준",
    "#### 11번가 반품/교환 이용방법",
    "##### 반품절차", "##### 교환절차",
    "#### 판매자정보", "판매자정보 테이블",
    "#### 구매시 주의사항",
    "11번가 지식재산권보호센터", "11번가 안전거래센터", "11번가 위해상품정보검색",
    "사이버범죄 예방정보 안내",
    "## 옵션 선택 및 주문하기",
    "총 0개", "적용 가능한 쿠폰 없음", "* 적용 가능한 쿠폰 없음", "[쿠폰변경](javascript:)",
    ":   총 상품금액에 배송비는 포함 되어 있지 않습니다.",
    ":   상품을 **장바구니**에 담았습니다.", ":   상품을 장바구니에 담았습니다.",
    "[구매하기](javascript:)", "[선물](javascript:)",
    "데이터를 로딩중입니다.",
    "## 장바구니에 담기", "장바구니에 담았습니다.", "장바구니 바로가기",
    "## 개인정보 수집 및 이용 동의",
    "우주패스ONLY 상품입니다.", "우주패스 가입자만 구매 가능합니다.", "(패밀리 멤버는 구매불가)",
    "[우주패스 시작하기](https://universepass.11st.co.kr)",
    "## 추가 혜택",
    "## 온누리상품권 사용안내", "온누리상품권",
    "## 배송 안내",
})

_ELEVENST_BOILERPLATE_SUBSTRINGS = (
    "11pay 포인트 적립 대상 상품을",
    "판매자 추가 적립 : 11pay 포인트",
    "11pay 포인트는 11번가",
    "적립된 11pay 포인트는 주문서 결제 시",
    "일부 카테고리 및 서비스는 사용 및 적립이 제한됩니다",
    "본문 50자 이상 작성 시",
    "검수 후 적합 여부에 따라 리뷰 건당",
    "결제금액이 3,000원 미만인 경우",
    "11pay 신한은행 계좌이체 결제 시",
    "최대 적립 포인트는 11pay 포인트 적립",
    "11번가 신한카드 결제 시 적립 포인트는",
    "정확한 적립 포인트는 결제 페이지에서",
    "PC 바로가기ON, 11번가앱에서 결제한 경우",
    "타 사이트를 통해 방문 시",
    "적립금액은 개인별 적립 한도에 따라",
    "제공된 비트코인 리워드는",
    "구매확정일로부터 2일 후",
    "디지털온누리상품권 결제 가능",
    "온누리상품권으로 결제 시",
    "결제 시 디지털온누리 앱 설치가 필요",
    "상품 수령 후 7일 이내에 신청하실 수 있습니다",
    "추가적으로 다음에 해당하는 반품/교환은 신청이 불가능할 수 있습니다",
    "소비자의 책임 있는 사유로 상품 등이 멸실",
    "소비자의 사용 또는 소비에 의해 상품 등의 가치가",
    "시간 경과에 의해 재판매가 곤란할 정도로",
    "복제가 가능한 상품 등의 포장을 훼손한 경우",
    "소비자의 주문에 따라 개별적으로 생산되는",
    "고객 귀책 사유 (단순 변심",
    "다른 옵션 상품으로 교환을 요청하는 경우",
    "슈팅셀러 상품의",
    "슈팅배송 상품의 교환을 요청하는 경우",
    "나의 11번가 주문내역에서",
    "원하는 상품 '반품신청' 클릭",
    "사유와 수거지 입력",
    "배송업체에서 상품 수거",
    "판매자 확인 후 반품 처리 완료",
    "원하는 상품의",
    "'교환신청' 클릭",
    "교환 접수 사유와 수거지 입력",
    "판매자 확인 후 교환할",
    "상품을 고객에게 발송",
    "교환 상품 배송 완료 후",
    "교환 처리 완료",
    "11번가 결제대금예치업 등록번호",
    "전자금융거래법에 따라",
    "구매금액, 결제수단에 상관없이",
    "11번가 공식 사이트(11st.co.kr) 외 피싱",
    "전자상거래 등에서의 소비자보호에 관한 법률",
    "미성년자가 물품을 구매하는 경우",
    "인증대상 상품을 구매하실 경우",
    "11번가의 결제시스템을 이용하지 않고",
    "등록된 판매물품과 내용은 판매자가 등록한 것으로",
    "지식재산권 보호를 위해",
    "본인의 지식",
    "재산권을 침해한 상품이 있을 시",
    "소비자보호를 위해 안전",
    "거래센터를 운영하고 있습니다",
    "안전거래와 관련된 궁금한 사항이",
    "안전한 상품 판매를 위해",
    "상품정보를 제공하고 있습니다",
    "11번가에서 걱정없이 편리하고",
    "우주패스 가입 시",
    "동일 묶음배송",
    "도서산간 추가 배송비",
    "판매자, 택배사 사정으로 예측치와 다를 수 있습니다",
)


def _strip_elevenst_boilerplate(text: str) -> str:
    return _strip_lines(text, _ELEVENST_BOILERPLATE_LINES, _ELEVENST_BOILERPLATE_SUBSTRINGS)


def _strip_boilerplate(text: str, url: str) -> str:
    """URL의 도메인에 맞는 boilerplate 제거 함수로 분기한다. 모르는 도메인은
    그대로 둔다(사이트마다 페이지 구조가 달라 걸러도 되는지 확신 없는 도메인의
    텍스트를 잘못 건드리는 것보다, 안 건드리고 자르기만 하는 기존 동작이 낫다)."""
    host = urlsplit(url).netloc.lower()
    if host == "danawa.com" or host.endswith(".danawa.com"):
        return _strip_danawa_boilerplate(text)
    if host == "11st.co.kr" or host.endswith(".11st.co.kr"):
        return _strip_elevenst_boilerplate(text)
    return text

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
        text = _strip_boilerplate(raw, r["url"])[:1500] if raw else snippet
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
    return _strip_boilerplate(raw, url) if raw else raw


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
