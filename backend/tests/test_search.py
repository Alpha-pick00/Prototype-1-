"""app/search.py의 그라운딩 소프트 신호 검색(search_coupang/search_naver) 테스트.
네트워크 요청 금지 - _tavily_search를 직접 monkeypatch해서 호출 인자만 검증한다
(실제 httpx 동작은 danawa 어댑터 테스트들처럼 이미 검증된 패턴이라 여기서 다시 안 봄)."""

from __future__ import annotations

import asyncio

import httpx

from app import search as search_module
from app.schemas import SearchResult


# -- app.search._strip_danawa_boilerplate -------------------------------------
# 실측 raw_content 샘플 기반(2026-08-19: "나이키 에어포스1"/"설화수 자음생크림"
# 검색에서 캡처) - 다나와 페이지 공통 네비게이션/공유버튼/푸터 문구를 걸러내고
# 상품마다 달라지는 실제 정보(상세 스펙 등)는 남기는지 확인한다.

_SNEAKER_RAW_SAMPLE = """메인 메뉴로 바로가기 본문으로 바로가기
 에누리
 몰테일
 메이크샵
나이키 에어포스 1 07 DD8959-100 (공식판매처) : 다나와 가격비교
다나와 가격비교 CI
최근
로그인
 로그인
 회원가입
 마이페이지
 쪽지
 광고센터
 고객센터
컨텐츠 상단으로 이동   컨텐츠 하단으로 이동
로딩중
나이키 에어포스 1 07 DD8959-100 (공식판매처) 상품비교
상세 스펙
:   운동화 / 여성용(W) / 로우탑 / 색상: 화이트 / 출시가: 139,000원
 관심
 공유
공유하기
레이어 닫기
+ 카카오톡
+ 라인
+ 페이스북
+ X
+ 밴드
 복사
URL이 복사되었습니다.
원하는 곳에 붙여넣기(Ctrl+V)하세요.
레이어 닫기
 신고
 인쇄
동영상 재생
나이키 에어포스 1 07 DD8959-100 (공식판매처)_이미지"""


def test_strip_danawa_boilerplate_removes_known_chrome_lines():
    cleaned = search_module._strip_danawa_boilerplate(_SNEAKER_RAW_SAMPLE)

    for chrome in ("로그인", "회원가입", "다나와 가격비교 CI", "공유하기", "레이어 닫기", "인쇄", "신고"):
        assert chrome not in cleaned.split("\n")


def test_strip_danawa_boilerplate_keeps_actual_product_spec_line():
    cleaned = search_module._strip_danawa_boilerplate(_SNEAKER_RAW_SAMPLE)

    assert "운동화 / 여성용(W) / 로우탑 / 색상: 화이트 / 출시가: 139,000원" in cleaned


def test_strip_danawa_boilerplate_removes_image_alt_text_lines():
    cleaned = search_module._strip_danawa_boilerplate(_SNEAKER_RAW_SAMPLE)

    assert "나이키 에어포스 1 07 DD8959-100 (공식판매처)_이미지" not in cleaned


def test_strip_danawa_boilerplate_drastically_shrinks_boilerplate_heavy_text():
    """실측(2026-08-19): 원본 1500자 중 상품 정보는 한 줄뿐이었다 - 걸러낸 뒤
    길이가 원본보다 뚜렷하게 짧아져야 한다(회귀 감지용, 정확한 비율은 안 박음)."""
    cleaned = search_module._strip_danawa_boilerplate(_SNEAKER_RAW_SAMPLE)

    assert len(cleaned) < len(_SNEAKER_RAW_SAMPLE) * 0.5


def test_strip_danawa_boilerplate_passes_through_unknown_text_unchanged():
    """모르는 텍스트(다나와 boilerplate 목록에 없는 줄)는 안전하게 그대로 둔다."""
    text = "이것은 임의의 상품 설명 텍스트입니다.\n가격은 12,900원입니다."

    cleaned = search_module._strip_danawa_boilerplate(text)

    assert "이것은 임의의 상품 설명 텍스트입니다." in cleaned
    assert "가격은 12,900원입니다." in cleaned


def test_search_coupang_scopes_tavily_to_coupang_domain(monkeypatch):
    captured: dict = {}

    async def _fake_tavily_search(query, max_results, domains=None):
        captured["query"] = query
        captured["max_results"] = max_results
        captured["domains"] = domains
        return [SearchResult(title="쿠팡 상품", url="https://coupang.com/vp/products/1", snippet="...")]

    monkeypatch.setattr(search_module, "_tavily_search", _fake_tavily_search)

    results = asyncio.run(search_module.search_coupang("무선 이어폰"))

    assert captured["query"] == "무선 이어폰"
    assert captured["domains"] == search_module.COUPANG_DOMAINS
    assert len(results) == 1
    assert results[0].url == "https://coupang.com/vp/products/1"


def test_search_coupang_returns_empty_list_on_failure(monkeypatch):
    async def _boom(query, max_results, domains=None):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(search_module, "_tavily_search", _boom)

    results = asyncio.run(search_module.search_coupang("무선 이어폰"))

    assert results == []


def test_search_naver_scopes_tavily_to_naver_shopping_domain(monkeypatch):
    captured: dict = {}

    async def _fake_tavily_search(query, max_results, domains=None):
        captured["query"] = query
        captured["max_results"] = max_results
        captured["domains"] = domains
        return [SearchResult(title="네이버 상품", url="https://shopping.naver.com/products/1", snippet="...")]

    monkeypatch.setattr(search_module, "_tavily_search", _fake_tavily_search)

    results = asyncio.run(search_module.search_naver("무선 이어폰"))

    assert captured["query"] == "무선 이어폰"
    assert captured["domains"] == search_module.NAVER_DOMAINS
    assert len(results) == 1
    assert results[0].url == "https://shopping.naver.com/products/1"


def test_search_naver_returns_empty_list_on_failure(monkeypatch):
    async def _boom(query, max_results, domains=None):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(search_module, "_tavily_search", _boom)

    results = asyncio.run(search_module.search_naver("무선 이어폰"))

    assert results == []


def test_search_unrestricted_passes_empty_domains_list(monkeypatch):
    captured: dict = {}

    async def _fake_tavily_search(query, max_results, domains=None):
        captured["query"] = query
        captured["max_results"] = max_results
        captured["domains"] = domains
        return [SearchResult(title="어딘가의 리뷰 글", url="https://example.com/review", snippet="...")]

    monkeypatch.setattr(search_module, "_tavily_search", _fake_tavily_search)

    results = asyncio.run(search_module.search_unrestricted("희귀 상품명"))

    assert captured["query"] == "희귀 상품명"
    assert captured["domains"] == []
    assert len(results) == 1
    assert results[0].url == "https://example.com/review"


def test_search_unrestricted_returns_empty_list_on_failure(monkeypatch):
    async def _boom(query, max_results, domains=None):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(search_module, "_tavily_search", _boom)

    results = asyncio.run(search_module.search_unrestricted("희귀 상품명"))

    assert results == []


def test_tavily_search_filters_out_generic_listing_pages(monkeypatch):
    """다나와 카테고리 목록 페이지처럼 특정 상품 하나를 가리키지 않는 결과는
    propose에게 넘기기 전에 걸러야 한다(2026-08-19 사용자 리포트: "10만원대
    이어폰 추천해줘 했는데 아무것도 안뜨잖아" - "이어폰" 검색 결과가 목록
    페이지로 뒤덮여 propose가 실제 상품을 하나도 못 봤다)."""
    fixture_response = {
        "results": [
            {
                "title": "무선 이어폰 : 다나와 가격비교",
                "url": "https://prod.danawa.com/list?cate=12237349",
                "content": "카테고리 목록",
            },
            {
                "title": "QCY Mini 2 : 다나와 가격비교",
                "url": "https://prod.danawa.com/info?pcode=6833593",
                "content": "39,900원",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture_response)

    real_async_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    monkeypatch.setattr(search_module.httpx, "AsyncClient", factory)

    results = asyncio.run(search_module._tavily_search("이어폰", 5))

    assert len(results) == 1
    assert results[0].url == "https://prod.danawa.com/info?pcode=6833593"


def test_tavily_search_strips_boilerplate_from_raw_content_before_truncating(monkeypatch):
    """raw_content가 있으면 앞 1500자를 그냥 자르는 대신, boilerplate를 먼저
    걷어낸 뒤 자른다 - 그래야 잘려나가는 뒷부분에 있던 실제 상품 정보(상세 스펙
    등)가 살아남는다(2026-08-19 실측: 앞 1500자는 거의 전부 로그인/메뉴 문구)."""
    fixture_response = {
        "results": [
            {
                "title": "나이키 에어포스 1 07 : 다나와 가격비교",
                "url": "https://prod.danawa.com/info?pcode=1",
                "content": "짧은 스니펫",
                "raw_content": _SNEAKER_RAW_SAMPLE,
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture_response)

    real_async_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    monkeypatch.setattr(search_module.httpx, "AsyncClient", factory)

    results = asyncio.run(search_module._tavily_search("나이키 에어포스1", 5))

    assert len(results) == 1
    snippet = results[0].snippet
    assert "운동화 / 여성용(W) / 로우탑 / 색상: 화이트 / 출시가: 139,000원" in snippet
    assert "로그인" not in snippet.split("\n")
