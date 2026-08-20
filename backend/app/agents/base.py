import json
import re
from urllib.parse import urlsplit

from ..schemas import SearchResult

_GENERIC_LISTING_URL_PATTERN = re.compile(
    r"(search\?|dsearch\.php|/search\.|Gateway\.[a-z]+\?|/list\?cate="
    # 모바일 다나와의 카테고리 목록 페이지(예: m.danawa.com/product/
    # productList.html?cateCode=11252453) - 단일 상품 페이지(product.html?code=)와
    # 이름이 비슷해 헷갈리기 쉽지만 "List"가 붙고 파라미터도 cateCode라 별개다
    # (2026-08-19 실측: "이어폰" 검색 결과에 섞여 들어와 다른 목록 페이지들처럼
    # propose에게 아무 상품 정보도 못 준 채 프롬프트만 채웠다).
    r"|productList\.html\?"
    r"|[?&](q|query|prdid|code)=($|&|#))",
    re.IGNORECASE,
)


def is_generic_listing_url(url: str) -> bool:
    """검색/카테고리 목록 페이지처럼 특정 상품 하나를 가리키지 않는 URL인지 판별.
    프롬프트로 LLM에게 피하라고만 하면 종종 무시되므로, 응답을 받은 뒤 코드에서 한 번 더 걸러낸다."""
    return bool(_GENERIC_LISTING_URL_PATTERN.search(url))


def is_danawa_comparison_page(url: str) -> bool:
    """다나와 가격비교 페이지 자체를 가리키는 URL인지 판별한다(2026-08-16,
    그라운딩 회귀 파일럿에서 발견: "위닉스 뽀송 제습기 16L"/"스캇 자전거
    헬멧" 검색에서 Qwen·DeepSeek이 독립적으로 이 페이지 자체를 후보로
    제안하면서 retailer를 "다나와"(판매처가 아니라 가격비교 사이트 자신),
    price를 빈 문자열로 채웠다 - 이 페이지는 여러 판매처를 나열만 할 뿐
    특정 판매처로 바로 연결되지 않아 애초에 가격/판매처 정보가 있을 수
    없는 후보다. challenge가 상품 정체성 일치만 확인하고 "실제 구매 가능한
    판매처인가"는 안 봐서 verified=True로 통과되는 사례를 확인했다 -
    propose 단계에서 후보 자체를 아예 안 받아들이도록 막는다.

    (2026-08-17 확장) 처음엔 prod.danawa.com/info 경로만 정규식으로 걸렀는데,
    재검증 파일럿에서 같은 문제의 다른 변형(m.danawa.com/product/product.html,
    모바일 페이지)이 그대로 통과되는 걸 확인했다 - 이 앱 전체의 기존 원칙
    (app.price_table._is_price_comparison_domain/_is_danawa_bridge_passthrough)
    과 동일하게, "다나와 도메인이면서 /bridge/(실제 구매 리다이렉트) 경로가
    아닌 모든 페이지"로 일반화했다 - 특정 URL 모양을 하나씩 allowlist하는
    대신 이 앱의 도메인 판단 기준 자체를 재사용한다."""
    host = urlsplit(url).netloc.lower()
    if not (host == "danawa.com" or host.endswith(".danawa.com")):
        return False
    return "/bridge/" not in urlsplit(url).path


NO_CANDIDATE_ERROR = "적절한 상품 후보를 찾지 못했습니다."


def filter_candidates(items: list, max_items: int = 3) -> list[dict]:
    """product_name이나 url이 비어 있거나, url이 일반 목록 페이지인 후보는
    제외한다. url이 빈 후보를 그대로 통과시키면, 실제로 살 수 있는 페이지가
    없는 후보가 병합/심사까지 흘러가 judge가 존재하지 않는 URL을 스스로
    지어내 채우는 문제가 생긴다 — 근거가 확실하지 않으면 애초에 후보를
    반환하지 말라고 프롬프트에도 적혀 있지만, 코드에서 한 번 더 걸러낸다.
    개수를 억지로 채우지 않고, 유효한 후보만 에이전트가 제시한 선호 순서 그대로
    최대 max_items개까지 남긴다."""
    filtered = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url or is_generic_listing_url(url) or is_danawa_comparison_page(url):
            continue
        if not (item.get("product_name") or "").strip():
            continue
        filtered.append(item)
        if len(filtered) >= max_items:
            break
    return filtered


PRICE_CONFIDENCE_GUIDANCE = (
    "price는 검색 결과 텍스트에 그 상품 자체의 가격으로 명확히 숫자가 나와 있을 때만 적으세요. "
    "다른 용량/수량/판촉가와 헷갈리거나 정확한 숫자를 확신할 수 없으면 "
    "실제와 다른 가격을 적느니 price를 빈 문자열로 두세요."
)

PRICE_KRW_GUIDANCE = (
    "price_krw는 검색 결과 텍스트에 그 상품 자체의 가격으로 명확히 숫자가 나와 있을 때만, "
    "쉼표나 '원' 같은 문자 없이 순수 정수로 적으세요. "
    "다른 용량/수량/판촉가와 헷갈리거나 정확한 숫자를 확신할 수 없으면 "
    "실제와 다른 가격을 적느니 price_krw를 null로 두세요."
)

PROPOSAL_INSTRUCTIONS = (
    "당신은 쇼핑 후보를 조사해 상품을 추천하는 에이전트입니다. "
    "아래 검색 결과를 참고해 사용자의 질의에 적합한 상품 후보를 최대 5개까지, "
    "당신이 가장 적합하다고 판단하는 순서대로 배열에 담아 반환하세요. "
    "근거가 확실한 후보가 1개뿐이면 1개만 반환하세요 — 개수를 채우려고 "
    "억지 후보를 만들어내지 마세요. 같은 상품을 중복해서 넣지 마세요. "
    "질의에 브랜드·용량·개수가 구체적으로 명시돼 있다면(예: '메로나 빙그레 70mL 10개') "
    "그 값과 명확히 일치하는 후보만 남기세요 — 브랜드만 맞고 용량이나 개수가 "
    "다른 상품(예: 70mL를 찾는데 80mL만 있는 상품)은 후보에서 제외하세요. "
    "질의에 명시되지 않은 항목은 자유롭게 골라도 됩니다. "
    "반드시 아래 JSON 배열 형식으로만 답하세요. 다른 텍스트나 코드펜스를 덧붙이지 마세요.\n\n"
    '[{"product_name": "...", "price_krw": 12900, "retailer": "...", '
    '"url": "...", "reasoning": "..."}, ...]\n\n'
    "url은 반드시 아래 검색 결과에 나온 URL을 그대로(수정 없이) 복사해서 쓰세요. "
    "검색 결과에 없는 URL을 새로 만들어내지 마세요. "
    "검색창/카테고리 목록 같은 일반 검색 페이지(예: query= 뒤가 비어있는 URL, "
    "검색 결과 전체 목록 페이지)는 후보에서 아예 제외하세요. "
    "특정 상품 하나를 가리키는 상세 페이지 URL을 가진 항목만 후보로 삼으세요. "
    f"{PRICE_KRW_GUIDANCE} "
    "검색 결과가 없거나 적절한 상품을 찾을 수 없으면 빈 배열 []을 반환하세요."
)


BULK_PROPOSAL_INSTRUCTIONS = (
    "당신은 쇼핑 후보를 조사해 서로 다른 브랜드의 후보를 제안하는 에이전트입니다. "
    "아래 검색 결과를 참고해 사용자가 사려는 상품과 일치하는, 서로 다른 브랜드의 후보를 "
    "최대 {max_options}개까지 찾아 각 브랜드의 최저가로 제시하세요. "
    "반드시 아래 JSON 배열 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '[{{"brand": "...", "product_name": "...", "price": "...", "retailer": "...", "url": "...", '
    '"reasoning": "...", "delivery_note": "..."}}, ...]\n\n'
    "reasoning은 이 브랜드에서 왜 이 상품을 최저가 후보로 골랐는지 한두 문장으로 설명하세요. "
    "delivery_note는 검색 결과 텍스트에 로켓배송/당일출고/익일배송처럼 배송 소요 정보가 "
    "명시적으로 나와 있을 때만 그 표현을 그대로 적고, 없으면 빈 문자열로 두세요(지어내지 마세요). "
    "url은 반드시 아래 검색 결과에 나온 URL을 그대로(수정 없이) 복사해서 쓰세요. "
    "검색 결과에 없는 URL을 새로 만들어내지 마세요. "
    "검색창/카테고리 목록 같은 일반 검색 페이지(예: query= 뒤가 비어있는 URL, "
    "검색 결과 전체 목록 페이지)는 후보에서 아예 제외하세요. "
    "특정 상품 하나를 가리키는 상세 페이지 URL을 가진 항목만 후보로 삼으세요. "
    f"{PRICE_CONFIDENCE_GUIDANCE} "
    "검색 결과에서 적절한 브랜드를 찾을 수 없으면 빈 배열 []을 반환하세요."
)


# Tavily snippet은 실측으로 건당 최대 1500자까지 나온다 - 검색 결과 12건을 그대로
# 넣으면 프롬프트가 2만자(약 1만3천 토큰)를 넘어, Groq 무료(on-demand) 티어의
# 분당 토큰(TPM) 한도(6000~12000)를 매번 초과했다(2026-08-16, "gemini" 슬롯을
# Groq로 옮기며 발견). 상품 하나를 식별하는 데 스니펫 전체가 필요하진 않으므로
# 500자로 자른다 - 이 함수가 propose/challenge/clarify/브랜드가격 프롬프트에 전부
# 쓰여 Qwen/DeepSeek 쪽 비용도 같이 줄어든다.
_SNIPPET_MAX_CHARS = 500


def format_results_block(search_results: list[SearchResult]) -> str:
    def _snippet(text: str) -> str:
        return text if len(text) <= _SNIPPET_MAX_CHARS else text[:_SNIPPET_MAX_CHARS] + "…"

    return (
        "\n".join(f"- {r.title} ({r.url}): {_snippet(r.snippet)}" for r in search_results)
        or "(검색 결과 없음)"
    )


def build_prompt(query: str, search_results: list[SearchResult]) -> str:
    results_block = format_results_block(search_results)
    return f"{PROPOSAL_INSTRUCTIONS}\n\n사용자 질의: {query}\n\n검색 결과:\n{results_block}"


def build_bulk_prompt(query: str, search_results: list[SearchResult], max_options: int = 5) -> str:
    results_block = format_results_block(search_results)
    instructions = BULK_PROPOSAL_INSTRUCTIONS.format(max_options=max_options)
    return f"{instructions}\n\n사용자 질의: {query}\n\n검색 결과:\n{results_block}"


CLARIFY_ASK_INSTRUCTIONS = (
    "당신은 쇼핑을 도와주는 챗봇입니다. 사용자가 검색한 상품 중 아래 후보들 중 "
    "어떤 걸 찾는지 확인이 필요합니다. 후보 목록을 그대로 나열하거나 \"~를 "
    "선택하세요\" 같은 딱딱한 안내문 대신, 실제 상담원이 대화하듯 자연스러운 "
    "한두 문장으로 물어보세요 — 후보 중 몇 가지를 예시로 자연스럽게 언급해도 "
    "좋습니다. 매번 표현을 다르게 해서 기계적으로 반복하지 마세요. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{"message": "..."}'
)


def build_clarify_ask_prompt(query: str, options: list[str]) -> str:
    options_block = "\n".join(f"- {o}" for o in options)
    return f"{CLARIFY_ASK_INSTRUCTIONS}\n\n사용자 질의: {query}\n\n후보:\n{options_block}"


FACET_CLARIFY_INSTRUCTIONS = (
    "당신은 애매한 쇼핑 검색어를 몇 가지 기준(facet)으로 좁혀나가도록 돕는 에이전트입니다. "
    "사용자 질의가 여러 종류의 서로 다른 상품을 아우르는 넓은 카테고리 검색어라면"
    "(예: '음료수', '과자'), 아래 다나와 검색 결과 상품명들에 실제로 등장하는 정보를 "
    "바탕으로 사용자가 선택해 좁혀나갈 수 있는 기준을 최대 4개까지 뽑아주세요. 기준의 "
    "예시: 제조사, 브랜드, 시리즈, 모델, 제품분류, 기종, 케이스형태, 부가기능, "
    "패턴, 용량, 용기형태, 색상, 소재, 특징 - 이 상품군에 실제로 의미 있는 기준만 "
    "고르세요(예: 전자기기 본체는 시리즈/모델/용량이, 식음료/생활용품은 용기형태/특징이, "
    "케이스·커버·필름·거치대처럼 특정 기기에 맞춰 쓰는 액세서리는 제품분류/기종/"
    "케이스형태/부가기능/패턴이 더 유의미할 수 있습니다 - 이런 액세서리는 다나와/쿠팡 "
    "실제 상세검색 필터도 이 방식을 씁니다). "
    "'시리즈'/'모델'은 상품 자체가 곧 그 기기인 경우(스마트폰·노트북 등 본체)에만 "
    "쓰세요 - '기종'과 헷갈리지 마세요. "
    "'기종'(또는 '핸드폰 기종')은 상품 자체가 특정 기기의 부속품·액세서리일 때, 그 "
    "상품이 맞는 기기 모델을 뜻합니다(다나와/쿠팡에서 케이스·필름·거치대 카테고리가 "
    "실제로 쓰는 필터 방식). 이때 상품 자체의 세대·형태를 시리즈/모델에 넣지 말고 "
    "'기종'으로 분류하되, 갤럭시/아이폰처럼 기기 브랜드가 여러 개 섞여 있어도 나누지 "
    "말고 '기종' 기준 하나에 다 같이 넣으세요(사용자가 자기 기기 이름만 알면 되지, "
    "브랜드별로 나뉜 여러 기준을 오갈 필요는 없습니다). "
    "'제품분류'는 이 상품이 정확히 어떤 종류의 물건인지(예: 휴대폰케이스/케이스 "
    "액세서리/방수팩·케이스, 또는 스탠드/거치대/그립톡처럼 아예 다른 종류)를 뜻합니다 - "
    "케이스형태(아래)보다 한 단계 위의 큰 분류입니다. "
    "'케이스형태'는 범퍼케이스/젤리케이스/하드케이스/다이어리케이스/카드수납케이스/"
    "플립커버형/바형/파우치형/방수케이스처럼 여러 브랜드에 공통되는 물리적 형태·구조만을 "
    "뜻합니다 - 브랜드 하나에서만 쓰는 고유 마케팅 제품라인명(예: 특정 브랜드만의 "
    "시리즈 이름)은 다른 상품과 비교해 좁혀나가는 데 쓸모가 없으니 어떤 기준에도 "
    "넣지 말고 아예 버리세요. "
    "'부가기능'은 맥세이프/마그네틱/무선충전가능/카드수납/스탠드겸용처럼 상품에 딸린 "
    "기능성 속성만을 뜻합니다. "
    "'패턴'은 클리어/투명/캐릭터/스트라이프/도트/로고·문구/플라워/애니멀/그래픽/체크처럼 "
    "디자인·무늬만을 뜻합니다 - 부가기능(기능)이나 케이스형태(구조)와 겹치는 값을 "
    "패턴에 다시 넣지 마세요. "
    "'용기형태'는 페트/캔/유리병/파우치/박스처럼 상품을 담는 물리적 용기의 재질·형태만을 "
    "뜻합니다 - '업소용'/'가정용'/'벌크'/'낱개'/'묶음'/'세트'처럼 누가 어떤 방식으로 "
    "구매하는지를 나타내는 값은 용기형태가 아니라 '구매유형' 기준으로 따로 분류하세요. "
    "'구매유형'은 정품/리퍼/중고/전시품/병행수입/해외구매/구매대행처럼 상품 "
    "상태나 유통 경로를 뜻합니다 - 이 기준은 상품명에 그 근거가 되는 단어가 실제로 "
    "등장할 때만 만드세요. 모델명, 시리즈명, 색상, 용량처럼 다른 기준에 속하는 값을 "
    "'구매유형'에 넣지 마세요. 또한 검색어가 특정 상품 카테고리(예: 스마트폰 같은 "
    "전자기기)에서 그 카테고리 시장에 통상 '해외구매'/'중고' 매물이 많다는 사전 지식만으로 "
    "이 기준을 만들지 마세요 - 실제 상품명 목록에 그런 단어가 없으면 만들면 안 됩니다. "
    "'특징'은 색상/소재/디자인/방수/충격방지/자석내장/카드수납/스트랩홀처럼 상품명에 "
    "실제로 등장하는 구체적인 기능·재질 속성만을 뜻합니다(위에 더 구체적인 기준인 "
    "부가기능/패턴/케이스형태가 있으면 그쪽을 우선하고, '특징'은 그 어디에도 안 맞는 "
    "나머지 속성에만 쓰세요) - 기종/시리즈/모델/용량/브랜드/케이스형태/부가기능/패턴 "
    "기준에 이미 넣은 값과 같은 정보를 '특징'에 중복해서 다시 넣지 마세요. 같은 값을 "
    "여러 기준에 동시에 넣지 마세요 - 값 하나는 가장 알맞은 기준 하나에만 속해야 "
    "합니다. 또한 어떤 기준 안에서 한 값이 같은 기준의 다른 값에 이미 완전히 포함되는 "
    "부분 문자열이면(예: '갤럭시S25 울트라'가 이미 있는데 '울트라'만 따로 또 값으로 "
    "넣는 경우) 그 부분 문자열 값은 넣지 마세요. "
    "케이스·커버·필름·거치대 같은 액세서리 상품이면 제품분류/기종/케이스형태/부가기능/"
    "패턴처럼 더 구체적인 기준을 '브랜드'/'제조사'보다 우선하세요 - 이런 구체적인 "
    "기준만으로 이미 4개가 채워지면 '브랜드'는 굳이 넣지 않아도 됩니다. "
    "'브랜드'나 '제조사' 기준을 넣을 때는 상품명에 실제로 등장하는 서로 다른 "
    "브랜드를 최대 15개까지(적으면 있는 만큼만) 뽑고, 그 외 기준은 최대 6개까지 뽑아주세요. "
    "각 기준 안에서는 상품명 목록에 더 많이 등장하는(더 인기 있는) 값부터 먼저 오도록 "
    "순서대로 나열하세요. 상품명 전부가 사실상 같은 값 하나뿐인 기준(예: '핸드폰'을 "
    "검색했는데 카테고리가 전부 '스마트폰'인 경우)은 골라도 아무것도 안 좁혀지니 "
    "만들지 마세요 - 최소 서로 다른 값 2개 이상 있는 기준만 포함하세요. "
    "사용자 질의에 이미 등장한 개념과 같은 뜻인 값(예: 질의에 '텀블러'가 있는데 "
    "값으로 '텀블러'를, 질의에 '반팔티'가 있는데 값으로 '반팔 티셔츠'를, 질의에 "
    "'섬유유연제'가 있는데 값으로 '섬유유연제'를 넣는 경우 - 표현이 정확히 같지 "
    "않아도 뜻이 같으면 포함)은 사용자가 이미 답한 것이므로 어떤 기준에도 넣지 "
    "마세요. 그 값을 뺀 뒤 그 기준에 서로 다른 값이 2개 미만 남으면 그 기준 "
    "자체를 만들지 마세요. "
    "검색어가 이미 충분히 구체적인 상품 하나를 가리키고 있어 더 좁힐 필요가 없으면 "
    '빈 객체 {"facets": {}}를 반환하세요. '
    "'카테고리'(또는 '분류'/'종류'처럼 이 상품이 어떤 대분류에 속하는지를 묻는 기준)는 "
    "만들지 마세요 - 사용자는 이미 검색어로 자신이 찾는 상품 종류를 밝혔으므로, 이를 "
    "다시 확인하는 기준은 불필요합니다. 그보다 구체적인 기준(브랜드/시리즈/모델/용량 등)만 "
    "뽑으세요. "
    "실제로 아래 상품명들에 나온 값만 사용하고 지어내지 마세요. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트나 코드펜스를 덧붙이지 마세요.\n\n"
    '{"facets": {"브랜드": ["...", "..."], "용량": ["...", "..."]}}'
)


def build_facet_clarify_prompt(query: str, product_names: list[str]) -> str:
    names_block = "\n".join(f"- {n}" for n in product_names) or "(검색 결과 없음)"
    return f"{FACET_CLARIFY_INSTRUCTIONS}\n\n사용자 질의: {query}\n\n다나와 검색 결과 상품명:\n{names_block}"


def build_facet_clarify_prompt_for_labels(
    query: str, product_names: list[str], labels: list[str]
) -> str:
    """브랜드별 facet 보강(app.debate._enrich_facets_per_brand, 2026-08-13:
    "APLLE 을 선택했을때 시리즈 후보가 너무 적어") 전용 - 브랜드 하나로 이미
    좁힌 상품명만 다시 보여주고, 라벨은 자유롭게 고르게 두지 않고 이미 정해진
    라벨 그대로만 쓰라고 강제한다. 안 그러면 같은 개념도 호출마다 "시리즈"/
    "모델"처럼 다르게 이름 붙여서, 나중에 라벨로 병합할 때 못 맞춘다."""
    names_block = "\n".join(f"- {n}" for n in product_names) or "(검색 결과 없음)"
    labels_block = ", ".join(labels)
    instructions = (
        "당신은 이미 특정 브랜드로 좁혀진 상품명 목록에서, 정해진 기준(facet)별로 "
        f"선택지를 뽑는 에이전트입니다. 반드시 다음 라벨만 그대로 써서 답하세요(새 "
        f"라벨을 만들거나 이름을 바꾸지 마세요): {labels_block}. 각 라벨마다 아래 "
        "상품명들에 실제로 등장하는 서로 다른 값을 최대 6개까지, 더 많이 등장하는 "
        "값부터 먼저 오도록 뽑으세요. 그 라벨에 해당하는 값이 상품명에 없으면 그 "
        "라벨은 아예 빼세요(빈 배열 넣지 마세요). '용기형태' 라벨에는 페트/캔/유리병/"
        "파우치/박스처럼 물리적 용기의 재질·형태만 넣고, '업소용'/'가정용'/'벌크'/"
        "'낱개'/'묶음'/'세트'처럼 구매 방식을 나타내는 값은 넣지 마세요. '구매유형' "
        "라벨에는 정품/리퍼/중고/전시품/병행수입/해외구매/구매대행처럼 상품 상태나 "
        "유통 경로를 나타내는 값만 넣고, 그 근거 단어가 상품명에 실제로 없으면 "
        "만들지 마세요(카테고리 시장 관행에 대한 사전 지식만으로 만들지 마세요). "
        "'기종'/'호환기종'/'핸드폰 기종' 라벨에는 이 상품이 맞는 기기 모델(예: "
        "갤럭시S25, 아이폰15)만 넣고, 기기 브랜드가 여러 개 섞여 있어도 나누지 말고 "
        "그 라벨 하나에 다 같이 넣으세요. "
        "'제품분류' 라벨에는 이 상품이 정확히 어떤 종류인지(예: 휴대폰케이스/스탠드/"
        "거치대/그립톡)만 넣고, "
        "'케이스형태' 라벨에는 범퍼/젤리/하드/다이어리/카드수납/방수/톡(스탠드)처럼 "
        "여러 브랜드에 공통되는 물리적 형태만 넣으세요 - 브랜드 하나에서만 쓰는 고유"
        "제품라인명은 어떤 라벨에도 넣지 말고 버리세요. "
        "'부가기능' 라벨에는 맥세이프/마그네틱/무선충전가능/카드수납/스탠드겸용처럼 "
        "기능성 속성만 넣고, '패턴' 라벨에는 클리어/투명/캐릭터/스트라이프/도트/"
        "로고·문구/플라워/애니멀/그래픽/체크처럼 디자인·무늬만 넣으세요. "
        "'특징' 라벨에는 위 라벨 어디에도 안 맞는 나머지 색상/소재/디자인 속성만 "
        "넣고, 다른 라벨과 같은 정보를 중복해서 넣지 마세요. 같은 값을 여러 라벨에 "
        "동시에 넣지 마세요. "
        "실제로 상품명에 나온 값만 쓰고 "
        "지어내지 마세요. 반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트나 "
        "코드펜스를 덧붙이지 마세요.\n\n"
        f'{{"facets": {{"{labels[0] if labels else "..."}": ["...", "..."]}}}}'
    )
    return f"{instructions}\n\n사용자 질의: {query}\n\n다나와 검색 결과 상품명:\n{names_block}"


BRAND_PRICE_INSTRUCTIONS = (
    "당신은 검색 결과에서 특정 브랜드 상품의 최저가를 찾는 에이전트입니다. "
    '아래 검색 결과에서 "{brand}" 브랜드의 상품 중 가장 저렴한 것 하나만 찾아 '
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{{"product_name": "...", "price": "...", "retailer": "...", "url": "..."}}\n\n'
    "url은 반드시 검색 결과에 나온 URL을 그대로(수정 없이) 복사해서 쓰세요. "
    "검색 결과에 없는 URL을 새로 만들어내지 마세요. "
    "검색창/카테고리 목록 같은 일반 검색 페이지(예: query= 뒤가 비어있는 URL, "
    "검색 결과 전체 목록 페이지)는 후보에서 아예 제외하세요. "
    "특정 상품 하나를 가리키는 상세 페이지 URL을 가진 항목만 후보로 삼으세요. "
    f"{PRICE_CONFIDENCE_GUIDANCE} "
    '"{brand}" 브랜드의 상품 상세 페이지가 검색 결과에 없으면 모든 필드를 빈 문자열로 두세요.'
)


def build_brand_price_prompt(query: str, brand: str, search_results: list[SearchResult]) -> str:
    results_block = format_results_block(search_results)
    instructions = BRAND_PRICE_INSTRUCTIONS.format(brand=brand)
    return f"{instructions}\n\n사용자 질의: {query}\n\n검색 결과:\n{results_block}"


# 완전 일치 후보가 하나도 없을 때의 폴백 경로(2026-08-15, "적절한 상품 후보를
# 찾지 못하면 다시 fallback해서 feedback 구조로 돌아가서 가장 관련성 높은
# 상품을 추천해주는 시스템") - PROPOSAL_INSTRUCTIONS는 브랜드/스펙이 정확히
# 일치하지 않으면 후보를 아예 비워서 반환하도록 요구한다
# (그라운딩 - 존재하지 않는 상품을 지어내지 않기 위함). 이 프롬프트는 딱 그
# 엄격함만 완화해 "정확히 일치하진 않아도 검색 결과 중 가장 관련성 높은 것
# 하나"를 고르게 한다 - 여전히 실제로 검색 결과에 있는 상품만 골라야 하고
# (지어내기는 금지), 왜 완벽히 일치하지 않는지를 reasoning에 반드시 밝히게
# 해서 UI가 "낮은 확신" 캐비어로 그대로 보여줄 수 있게 한다.
RELAXED_PICK_INSTRUCTIONS = (
    "당신은 쇼핑 검색을 돕는 에이전트입니다. 아래 검색 결과 중 사용자 질의와 "
    "정확히 일치하는 상품이 없더라도, 실망시키지 않도록 그나마 가장 관련성 "
    "높은 상품 하나를 대신 추천해야 합니다. "
    "실제로 검색 결과에 나온 상품만 고르세요 - 존재하지 않는 상품을 지어내지 "
    "마세요. product_name/price/retailer/url은 그 검색 결과에 있는 값을 "
    "그대로 옮기세요. "
    "reasoning에는 반드시 이 상품이 질의와 정확히 일치하지 않는 이유(예: "
    "브랜드는 다르지만 같은 종류의 상품, 또는 용량/사양이 다름)를 솔직하게 "
    "먼저 밝히고, 그럼에도 가장 관련성 높다고 판단한 근거를 이어서 쓰세요. "
    "검색 결과 전체에 사용자가 찾는 것과 아예 다른 카테고리 상품만 있어서 "
    "추천할 만한 게 정말 하나도 없으면, 모든 필드를 빈 문자열로 두세요. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{"product_name": "...", "price": "...", "retailer": "...", "url": "...", "reasoning": "..."}'
)


def build_relaxed_pick_prompt(query: str, search_results: list[SearchResult]) -> str:
    results_block = format_results_block(search_results)
    return f"{RELAXED_PICK_INSTRUCTIONS}\n\n사용자 질의: {query}\n\n검색 결과:\n{results_block}"


REFINE_QUERY_INSTRUCTIONS = (
    "당신은 사용자의 쇼핑 검색어를 실제 쇼핑몰 검색에 더 유리하게 다듬는 에이전트입니다. "
    "질의가 이미 구체적이면(브랜드·모델명·용량 등이 명확하면) 그대로 반환하세요. "
    "모호한 표현(예: '그거', '요즘 유행하는')이 있으면 검색어에서 제거하고, "
    "질의에 이미 암시된 브랜드·스펙·용량이 있다면 명시적인 키워드로 풀어 쓰세요. "
    "질의가 '안녕 나 컵을 사고싶어', '안녕하세요 카메라 좀 추천해주세요'처럼 인사말이나 "
    "'~하고 싶어/~하려고요/~해주세요' 같은 대화체 문장으로 감싸져 있으면, 인사말·대명사·조사· "
    "구매 의도 표현을 전부 걷어내고 실제로 찾는 상품명(또는 상품 종류)만 검색어로 남기세요 "
    "(예: '안녕 나 컵을 사고싶어' → '컵'). "
    "검색어에 없는 브랜드나 상품 정보를 새로 지어내지 마세요 — 원래 의미를 벗어나면 안 됩니다. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{"query": "..."}'
)


def build_refine_query_prompt(query: str) -> str:
    return f"{REFINE_QUERY_INSTRUCTIONS}\n\n사용자 질의: {query}"


CHALLENGE_INSTRUCTIONS = (
    "당신은 다른 에이전트들이 제안한 쇼핑 후보를 비판적으로 검증하는 에이전트입니다. "
    "아래 후보 목록과 검색 결과를 비교해 각 후보를 두 기준으로 판단하세요: "
    "(1) 그라운딩 — 검색 결과 텍스트에 실제로 나오지 않는 가격/상품 정보를 지어내지 않았는지, "
    "(2) 정체성 일치 — 그 후보가 사용자 질의가 찾는 상품(브랜드·용량·개수를 포함한 스펙)과 "
    "실제로 일치하는지. 특히 질의에 용량이나 개수가 숫자로 구체적으로 적혀 있다면(예: "
    "'70mL 10개') 후보의 용량·개수가 그 숫자와 정확히 같은지 반드시 확인하세요 — "
    "브랜드와 상품명은 맞아도 용량이나 개수가 다르면(예: 70mL를 찾는데 후보가 80mL) "
    "verified를 false로 표시하고 note에 어떤 값이 다른지 적으세요. "
    "애매하거나 확신이 서지 않으면 verified를 false로 남발하지 말고, "
    "명백히 근거가 없거나 질의와 무관한 경우에만 false로 표시하세요. "
    "일부 후보에는 '실제 페이지 재조회 원문'이 함께 제공됩니다 — 이는 검색 당시 "
    "잘린 스니펫보다 더 최신이고 완전한 정보이므로, 스니펫과 내용이 다르면 "
    "재조회 원문을 우선 신뢰해 판단하세요. 재조회 원문에 이 후보와 명백히 같은 "
    "상품의 가격이 원 단위 숫자로 뚜렷이 나와 있고 그 값이 후보에 적힌 가격과 "
    "다르면, refreshed_price_krw에 재조회 원문 기준 가격을 정수로 적으세요(쉼표나 "
    "'원' 없이 숫자만) — 검색 당시보다 재조회 시점이 더 최신이라 가격이 바뀌었을 "
    "수 있습니다. 재조회 원문이 없거나, 있어도 이 후보의 가격을 명확히 특정할 수 "
    "없거나, 후보에 적힌 가격과 같으면 refreshed_price_krw는 생략하거나 null로 "
    "두세요 — 확신 없이 숫자를 지어내지 마세요. "
    "일부 경우에는 '쿠팡 교차 확인 검색 결과'나 '네이버쇼핑 교차 확인 검색 결과'가 "
    "추가로 제공됩니다 — 검색 결과와는 별도로 다른 쇼핑몰에서 같은 상품을 검색한 "
    "참고 자료입니다. 후보와 일치하는 상품이 그쪽에서도 보이면 그라운딩 신뢰도가 "
    "더 높다는 뜻으로 참고하세요. 둘 중 하나 또는 둘 다에 없다고 해서 곧바로 "
    "false로 판단하지 마세요(품절·검색 누락일 수 있음) — 원 검색 결과나 재조회 "
    "원문과 명백히 모순될 때만 이 신호를 근거로 우려를 표시하세요. "
    "반드시 입력된 후보와 같은 개수, 같은 순서로 아래 JSON 배열 형식으로만 답하세요. "
    "다른 텍스트나 코드펜스를 덧붙이지 마세요.\n\n"
    '[{"url": "...", "verified": true, "note": "...", "refreshed_price_krw": null}, ...]\n\n'
    "note는 검증 근거를 한 문장으로 설명하세요(통과든 우려든)."
)


def _format_candidates_block(candidates: list, candidate_pages: dict[str, str] | None = None) -> str:
    candidate_pages = candidate_pages or {}
    lines = []
    for i, c in enumerate(candidates, start=1):
        product_name = getattr(c, "product_name", None) or (c.get("product_name") if isinstance(c, dict) else None)
        price_krw = getattr(c, "price_krw", None) if not isinstance(c, dict) else c.get("price_krw")
        retailer = getattr(c, "retailer", None) if not isinstance(c, dict) else c.get("retailer")
        url = getattr(c, "url", None) if not isinstance(c, dict) else c.get("url")
        reasoning = getattr(c, "reasoning", None) if not isinstance(c, dict) else c.get("reasoning")
        page_text = candidate_pages.get(url) if url else None
        page_block = f"\n    실제 페이지 재조회 원문: {page_text[:2000]}" if page_text else ""
        lines.append(
            f"[{i}] 상품: {product_name} / 가격: {price_krw} / 판매처: {retailer} / "
            f"URL: {url} / 제안 근거: {reasoning}{page_block}"
        )
    return "\n".join(lines) or "(후보 없음)"


def build_challenge_prompt(
    query: str,
    candidates: list,
    search_results: list[SearchResult],
    candidate_pages: dict[str, str] | None = None,
    coupang_results: list[SearchResult] | None = None,
    naver_results: list[SearchResult] | None = None,
) -> str:
    candidates_block = _format_candidates_block(candidates, candidate_pages)
    results_block = format_results_block(search_results)
    prompt = (
        f"{CHALLENGE_INSTRUCTIONS}\n\n사용자 질의: {query}\n\n"
        f"검증할 후보:\n{candidates_block}\n\n검색 결과:\n{results_block}"
    )
    if coupang_results:
        prompt += f"\n\n쿠팡 교차 확인 검색 결과(참고용):\n{format_results_block(coupang_results)}"
    if naver_results:
        prompt += f"\n\n네이버쇼핑 교차 확인 검색 결과(참고용):\n{format_results_block(naver_results)}"
    return prompt


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def filter_bulk_options(options: list[dict], search_results: list[SearchResult]) -> list[dict]:
    """generic listing URL이거나 브랜드/상품명이 비어 있는 항목은 후보에서 제외한다.
    또한 brand가 실제로 그 url의 검색 결과(제목+본문)에 등장하지 않으면 제외한다 —
    검색 결과가 여러 개일 때 LLM이 다른 후보의 브랜드명을 엉뚱한 url에 붙이는
    매핑 오류가 가끔 있어, 응답을 받은 뒤 코드에서 한 번 더 근거를 확인한다."""
    url_text = {r.url: _normalize(r.title + r.snippet) for r in search_results}
    filtered = []
    for o in options:
        url = o.get("url") or ""
        brand = (o.get("brand") or "").strip()
        product_name = (o.get("product_name") or "").strip()
        if is_generic_listing_url(url) or not brand or not product_name:
            continue
        haystack = url_text.get(url)
        if haystack is not None and _normalize(brand) not in haystack:
            continue
        filtered.append(o)
    return filtered


def parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def parse_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response: {text[:200]!r}")
    return json.loads(match.group(0))
