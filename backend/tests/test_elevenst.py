"""fetchers.elevenst(11번가 오픈 API ProductSearch) 테스트.
네트워크 요청 금지 - parse_search_xml은 순수 함수로, search_elevenst는
httpx.MockTransport로 테스트한다."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from fetchers import elevenst

_SAMPLE_XML = """<?xml version="1.0" encoding="EUC-KR"?><ProductSearchResponse><Request><Arguments></Arguments></Request><Products><TotalCount>2</TotalCount><Product><ProductCode>111</ProductCode><ProductName><![CDATA[나이키 에어 포스 1 07]]></ProductName><ProductPrice>200000</ProductPrice><SalePrice>190000</SalePrice><SellerNick><![CDATA[퀀텍스]]></SellerNick><Seller>quantex</Seller><DetailPageUrl><![CDATA[http://www.11st.co.kr/product/SellerProductDetail.tmall?method=getSellerProductDetail&prdNo=111]]></DetailPageUrl><ReviewCount>5</ReviewCount><BuySatisfy>90</BuySatisfy></Product><Product><ProductCode>222</ProductCode><ProductName><![CDATA[나이키 에어 포스 1 07 화이트]]></ProductName><ProductPrice>180000</ProductPrice><SalePrice></SalePrice><SellerNick><![CDATA[]]></SellerNick><Seller>seller2</Seller><DetailPageUrl><![CDATA[http://www.11st.co.kr/product/SellerProductDetail.tmall?method=getSellerProductDetail&prdNo=222]]></DetailPageUrl><ReviewCount></ReviewCount><BuySatisfy></BuySatisfy></Product></Products></ProductSearchResponse>"""

_ERROR_XML = """<?xml version="1.0" encoding="EUC-KR"?><ProductSearchResponse><Error><Code>003</Code><Message>등록되지 않은 KEY 입니다.</Message></Error></ProductSearchResponse>"""


# -- parse_search_xml (순수 함수) ----------------------------------------------


def test_parse_search_xml_extracts_products():
    items = elevenst.parse_search_xml(_SAMPLE_XML)

    assert len(items) == 2
    assert items[0]["product_code"] == "111"
    assert items[0]["product_name"] == "나이키 에어 포스 1 07"
    assert items[0]["seller"] == "퀀텍스"
    assert items[0]["url"].endswith("prdNo=111")


def test_parse_search_xml_prefers_sale_price_over_list_price():
    items = elevenst.parse_search_xml(_SAMPLE_XML)

    assert items[0]["price_krw"] == 190000  # SalePrice(할인가) 우선


def test_parse_search_xml_falls_back_to_product_price_when_sale_price_empty():
    items = elevenst.parse_search_xml(_SAMPLE_XML)

    assert items[1]["price_krw"] == 180000  # SalePrice가 비어있어 ProductPrice로 대체


def test_parse_search_xml_falls_back_to_seller_id_when_nick_empty():
    items = elevenst.parse_search_xml(_SAMPLE_XML)

    assert items[1]["seller"] == "seller2"


def test_parse_search_xml_leaves_review_fields_none_when_blank():
    items = elevenst.parse_search_xml(_SAMPLE_XML)

    assert items[1]["review_count"] is None
    assert items[1]["buy_satisfy"] is None


def test_parse_search_xml_raises_on_error_response():
    with pytest.raises(elevenst.ElevenstSearchBlocked) as exc_info:
        elevenst.parse_search_xml(_ERROR_XML)

    assert exc_info.value.code == "003"


# -- search_elevenst 네트워크 래퍼 (httpx.MockTransport) -----------------------

_RealAsyncClient = httpx.AsyncClient


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def factory(**kwargs):
        return _RealAsyncClient(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    monkeypatch.setattr(elevenst.httpx, "AsyncClient", factory)


def test_search_elevenst_decodes_euc_kr_response_correctly(monkeypatch):
    """실측(2026-08-20)으로 찾은 함정 - 응답이 encoding="EUC-KR"로 온다.
    UTF-8로 잘못 디코딩하면 한글 필드가 전부 깨진다(mojibake) - 이 테스트가
    그 회귀를 감지한다."""
    monkeypatch.setattr("app.config.settings.elevenst_api_key", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_SAMPLE_XML.encode("euc-kr"))

    _patch_client(monkeypatch, handler)

    items = asyncio.run(elevenst.search_elevenst("나이키 에어포스1", limit=5))

    assert items[0]["product_name"] == "나이키 에어 포스 1 07"


def test_search_elevenst_returns_empty_list_when_api_key_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.elevenst_api_key", None)

    async def _boom(*_args, **_kwargs):
        raise AssertionError("API 키가 없는데 네트워크 호출이 나갔다")

    monkeypatch.setattr(elevenst.httpx, "AsyncClient", _boom)

    items = asyncio.run(elevenst.search_elevenst("아무거나"))

    assert items == []


def test_search_elevenst_respects_limit(monkeypatch):
    monkeypatch.setattr("app.config.settings.elevenst_api_key", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_SAMPLE_XML.encode("euc-kr"))

    _patch_client(monkeypatch, handler)

    items = asyncio.run(elevenst.search_elevenst("나이키 에어포스1", limit=1))

    assert len(items) == 1
