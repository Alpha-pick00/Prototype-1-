"""11번가 오픈 API(openapi.11st.co.kr, ProductSearch)로 상품을 검색하는 어댑터.

다나와(fetchers/danawa*.py)와 달리 HTML 스크래핑이 아니라 11번가가 공식
제공하는 구조화 XML API를 호출한다 - 페이지 구조가 사이트마다 달라 스니펫
파싱이 어긋나는 문제 자체가 없다.

실측(2026-08-20)으로 확인한 함정: 응답이 `encoding="EUC-KR"`로 온다(공식
가이드 문서에는 명시돼 있지 않음). 요청 키워드는 평범한 UTF-8 URL 인코딩으로
보내도 검색 자체는 정상 동작하지만(httpx가 자동으로 처리), 응답 바이트를
UTF-8로 디코딩하면 ProductName 등 모든 한글 필드가 깨진다 - 반드시
`response.content`를 `"euc-kr"`로 명시적으로 디코딩한 뒤 XML 파싱해야 한다.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict
from xml.etree import ElementTree

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

API_URL = "http://openapi.11st.co.kr/openapi/OpenApiService.tmall"
REQUEST_TIMEOUT = 10.0

# 실측(2026-08-20, apiCode=ProductSearch&option=Categories)으로 확인한 함정 -
# 11번가는 다나와식 대분류/중분류 상품 카테고리 색인이 아니라, 검색어와 매칭된
# "전시 카테고리"(dispCtgr) 목록을 평면으로 돌려준다. 여기엔 실제 상품
# 카테고리(예: "과자/간식")와 프로모션 탭/이벤트 전시관(예: "홈쇼핑 Tab",
# "9900원샵", "2020설", "쿠폰 발급용 추가 전시 카테고리")이 구분 없이 섞여
# 있고, 건수(CategoryPrdCnt)도 실제 상품 수가 아니라 그 전시관의 진열 슬롯
# 수라 조작된 상품 카테고리보다 더 높게 나오는 경우가 흔하다(실측: "오리온
# 초코파이 말차쇼콜라" 검색 시 "과자/간식"(117건, 실제 카테고리)보다 "홈쇼핑
# Tab"/"9900원샵"(각 120건, 프로모션 전시관)이 더 위에 옴). facet으로 그대로
# 노출하면 사용자에게 "카테고리: 홈쇼핑 Tab" 같은 무의미한 선택지가 뜨므로,
# 흔한 프로모션/이벤트 패턴을 이름으로 걸러낸다 - 완벽하지 않은 휴리스틱이라는
# 한계는 있다(11번가가 실제 상품 카테고리와 전시 카테고리를 API 응답에서
# 구분해 주지 않는 한 근본적으로 해소되지 않는다).
_PROMO_CATEGORY_NAME_RE = re.compile(
    r"(?:^\[\d{4}\]|Tab$|쿠폰|이벤트|전시|장바구니|기업서비스|폐점|^\d{4}\s*(?:년|설|럭셔리)|"
    r"홈쇼핑|샵$|마켓$|배송$|데이$)"
)


class ElevenstCategoryGroup(TypedDict):
    name: str
    count: int
    subcategories: list[dict]


class ElevenstSearchBlocked(RuntimeError):
    """API가 에러 코드로 응답했을 때만 던진다(예: 003 미등록 키) - 그 외
    실패(타임아웃, 파싱 실패, 결과 없음)는 빈 리스트로 조용히 처리한다는
    계약을 유지한다. fetchers.danawa_search.DanawaSearchBlocked와 같은 이유
    (배치 호출자가 "진짜로 상품이 없다"와 "키/설정이 잘못됐다"를 구분해야
    안전하게 멈출 수 있다)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"11번가 오픈 API 오류 (code={code}): {message}")
        self.code = code


class ElevenstSearchItem(TypedDict):
    product_code: str
    product_name: str
    price_krw: int
    seller: str
    url: str
    review_count: int | None
    buy_satisfy: int | None


def _text(el: ElementTree.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def parse_search_xml(xml_text: str) -> list[ElevenstSearchItem]:
    """네트워크 없이 순수하게 XML 응답만 파싱한다(fetchers.danawa_search의
    parse_search_html과 같은 패턴 - 테스트는 전부 이 함수를 통해서 한다).
    호출부가 이미 EUC-KR로 디코딩한 str을 넘겨준다는 전제."""
    root = ElementTree.fromstring(xml_text)

    error = root.find("Error")
    if error is not None:
        code = _text(error.find("Code"))
        message = _text(error.find("Message"))
        raise ElevenstSearchBlocked(code, message)

    items: list[ElevenstSearchItem] = []
    for product in root.findall(".//Product"):
        product_code = _text(product.find("ProductCode"))
        product_name = _text(product.find("ProductName"))
        url = _text(product.find("DetailPageUrl"))
        if not product_code or not product_name or not url:
            continue
        # SalePrice(할인 반영 실판매가)를 우선 쓰고, 없으면 ProductPrice(정가)로
        # 대체한다 - Benefit/Discount가 없는 상품은 SalePrice가 비어있을 수 있다.
        price_text = _text(product.find("SalePrice")) or _text(product.find("ProductPrice"))
        try:
            price_krw = int(price_text)
        except ValueError:
            continue
        review_text = _text(product.find("ReviewCount"))
        satisfy_text = _text(product.find("BuySatisfy"))
        items.append(
            ElevenstSearchItem(
                product_code=product_code,
                product_name=product_name,
                price_krw=price_krw,
                seller=_text(product.find("SellerNick")) or _text(product.find("Seller")) or "11번가",
                url=url,
                review_count=int(review_text) if review_text.isdigit() else None,
                buy_satisfy=int(satisfy_text) if satisfy_text.isdigit() else None,
            )
        )
    return items


async def search_elevenst(query: str, limit: int = 5) -> list[ElevenstSearchItem]:
    """11번가 ProductSearch API를 호출해 가격 오름차순(sortCd=A)으로 상품을
    찾는다. 키가 없으면(.env 미설정) 즉시 빈 리스트 - 호출부가 "설정 안 됨"과
    "검색 결과 없음"을 굳이 구분할 필요가 없는 초기 단계라 조용히 넘어간다."""
    if not settings.elevenst_api_key:
        return []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            API_URL,
            params={
                "key": settings.elevenst_api_key,
                "apiCode": "ProductSearch",
                "keyword": query,
                "pageNum": 1,
                "pageSize": limit,
                "sortCd": "A",
            },
        )
        response.raise_for_status()
        xml_text = response.content.decode("euc-kr")

    return parse_search_xml(xml_text)[:limit]


def parse_category_breakdown_xml(xml_text: str) -> list[ElevenstCategoryGroup]:
    """apiCode=ProductSearch&option=Categories 응답 끝에 붙는 <Categories>
    블록을 파싱한다(네트워크 없는 순수 함수 - 테스트는 이 함수로 한다).
    11번가는 2단계(대분류/중분류) 위계를 안 주고 평면 목록만 주므로
    subcategories는 항상 빈 리스트다 - _select_effective_category_name 등
    호출부가 다나와와 같은 모양(name/count/subcategories)을 기대해 구조만
    맞춰준다."""
    root = ElementTree.fromstring(xml_text)
    groups: list[ElevenstCategoryGroup] = []
    seen_names: set[str] = set()

    for category in root.findall(".//Categories/Category"):
        name = _text(category.find("CategoryName"))
        count_text = _text(category.find("CategoryPrdCnt"))
        if not name or not count_text.isdigit() or name in seen_names:
            continue
        if _PROMO_CATEGORY_NAME_RE.search(name):
            continue
        seen_names.add(name)
        groups.append(ElevenstCategoryGroup(name=name, count=int(count_text), subcategories=[]))

    return groups


async def search_categories(query: str) -> list[ElevenstCategoryGroup]:
    """검색어와 매칭되는 11번가 전시 카테고리 집계를 가져온다
    (app.debate.check_clarify_facets의 "카테고리" facet 실측 출처 -
    price_table._search_elevenst_categories가 이 함수를 감싼다). 키가 없으면
    빈 리스트."""
    if not settings.elevenst_api_key:
        return []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            API_URL,
            params={
                "key": settings.elevenst_api_key,
                "apiCode": "ProductSearch",
                "keyword": query,
                "option": "Categories",
                "pageSize": 1,
            },
        )
        response.raise_for_status()
        xml_text = response.content.decode("euc-kr")

    return parse_category_breakdown_xml(xml_text)
