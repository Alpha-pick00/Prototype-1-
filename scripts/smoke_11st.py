# -*- coding: utf-8 -*-
"""11번가 OpenAPI 상품검색 스모크 테스트.

목적: 아키텍처를 붙이기 전에 "11번가 API가 우리 16개 카테고리에서 실제로
쓸만한 결과(실존 상품 + 가격 + 구매링크)를 주는가"를 눈으로 확인한다.

사용법:
  1) 11번가 OpenAPI 키를 .env(프로젝트 루트나 backend/)에 넣거나 환경변수로:
       ELEVENST_API_KEY=발급받은키
     또는 실행 시 인자로:
       python scripts/smoke_11st.py 발급받은키
  2) python scripts/smoke_11st.py

카테고리별로 결과수/가격유무/링크유무를 요약해준다.
"""
import os
import sys
import xml.etree.ElementTree as ET

import requests

# 프로젝트 .env 로드 시도 (있으면)
try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))
except Exception:
    pass

API_KEY = (sys.argv[1] if len(sys.argv) > 1 else "") or os.environ.get("ELEVENST_API_KEY", "")
ENDPOINT = "http://openapi.11st.co.kr/openapi/OpenApiService.tmall"

# 16개 카테고리 커버리지를 보려고 일부러 다양하게 (가전만이 아니라 뷰티/패션/
# 반려동물/헬스/식품/스포츠까지). 11번가가 어느 카테고리에서 얇은지가 핵심 관찰점.
QUERIES = [
    ("가전디지털", "무선 이어폰"),
    ("식품(음료)", "네스프레소 호환 캡슐"),
    ("생활용품", "다우니 섬유유연제 대용량"),
    ("뷰티", "지성 피부 수분크림"),
    ("스포츠/레저", "요가매트"),
    ("반려동물용품", "소형견 사료"),
    ("헬스/건강식품", "오메가3"),
    ("패션의류/잡화", "남자 가을 코트"),
]

# 11st OpenAPI 응답 필드명이 문서/버전마다 조금씩 달라서(ProductName vs
# productName 등) 여러 후보 태그를 순서대로 시도한다.
NAME_TAGS = ["ProductName", "productName", "prdNm"]
PRICE_TAGS = ["ProductPrice", "productPrice", "SalePrice", "salePrice", "lwstPrc", "selPrc"]
URL_TAGS = ["DetailPageUrl", "detailPageUrl", "ProductImage", "productUrl", "detailPage"]
SELLER_TAGS = ["Seller", "seller", "sellerNick"]


def _first(el, tags):
    for t in tags:
        found = el.find(f".//{t}")
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def search(keyword, page_size=10):
    params = {
        "key": API_KEY,
        "apiCode": "ProductSearch",
        "keyword": keyword,
        "pageNum": 1,
        "pageSize": page_size,
    }
    r = requests.get(ENDPOINT, params=params, timeout=15)
    r.encoding = "euc-kr"  # 11st OpenAPI는 EUC-KR 인코딩 XML을 준다
    return r.text


def parse(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return None, f"XML 파싱 실패: {e} / {xml_text[:200]}"
    products = root.findall(".//Product") or root.findall(".//product")
    # 상품이 하나도 없을 때만 오류 태그를 확인한다(상품명에 '인증' 같은 단어가
    # 들어가도 오탐하지 않도록, 단순 문자열 포함이 아니라 실제 태그로 판정).
    if not products:
        err_el = root.find(".//ErrorCode") or root.find(".//errorCode") or root.find(".//message")
        if err_el is not None:
            return None, (err_el.text or "오류 응답")[:300]
    items = []
    for p in products:
        items.append(
            {
                "name": _first(p, NAME_TAGS),
                "price": _first(p, PRICE_TAGS),
                "url": _first(p, URL_TAGS),
                "seller": _first(p, SELLER_TAGS),
            }
        )
    return items, None


def main():
    if not API_KEY:
        print("❌ ELEVENST_API_KEY가 없습니다. .env에 넣거나 인자로 넘겨주세요.")
        print("   예) python scripts/smoke_11st.py YOUR_KEY")
        sys.exit(1)

    print(f"🔑 키 로드됨 (…{API_KEY[-4:]})\n")
    print(f"{'카테고리':<16}{'질의':<22}{'결과수':>5} {'가격':>4} {'링크':>4}  샘플")
    print("-" * 90)

    for cat, q in QUERIES:
        try:
            xml_text = search(q)
        except Exception as e:
            print(f"{cat:<16}{q:<22}  요청실패: {e}")
            continue
        items, err = parse(xml_text)
        if err:
            print(f"{cat:<16}{q:<22}  ⚠️ {err}")
            continue
        n = len(items)
        has_price = "✅" if any(i["price"] for i in items) else "❌"
        has_url = "✅" if any(i["url"] for i in items) else "❌"
        sample = ""
        if items:
            top = items[0]
            sample = f"{top['name'][:24]} / {top['price']}원 / {'링크O' if top['url'] else '링크X'}"
        print(f"{cat:<16}{q:<22}{n:>5} {has_price:>4} {has_url:>4}  {sample}")

    print("\n판단 기준: 대부분 카테고리에서 결과수>0 + 가격✅ + 링크✅ 면 → 하이브리드 붙일 만함.")
    print("특정 카테고리만 얇으면 → 그건 다나와가 커버하면 됨.")


if __name__ == "__main__":
    main()
