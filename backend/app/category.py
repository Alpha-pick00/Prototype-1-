"""검색 질의를 16개 대분류 카테고리로 분류한다.

clarify(Human-in-the-loop) 단계에서 브랜드/제품/용량/수량 4축을 모든 카테고리에
똑같이 물어보면, 그 카테고리엔 없는 축까지 선택지로 뜨는 문제가 있었다. 이
모듈은 LLM(Groq)으로 질의를 대분류 하나로 판별해, 용량·수량이 실제로 의미 있는
카테고리에서만 그 축을 clarify 옵션에 남기는 데 쓰인다.

용량(mL/L/g/kg)과 수량(묶음 개수)은 서로 독립된 축이라 따로 판단한다 —
intent.py의 단위 패턴을 보면 수량 쪽 단위(개/병/팩/박스/세트/캔/봉지/포/장/권/
벌/족/대)가 용량 쪽(ml/L/kg/g)보다 카테고리를 훨씬 넓게 가로지른다. 예를 들어
'권'은 도서, '벌'·'족'은 패션의류처럼 용량 개념이 아예 없는 카테고리에서도
수량만은 실제 구매 기준으로 쓰인다. 그래서 VOLUME_RELEVANT_CATEGORIES와
QUANTITY_RELEVANT_CATEGORIES를 별도 집합으로 둔다.

'식품' 대분류는 그 안에서도 편차가 커서 대분류 하나로만 켜고 끄기엔 부족하다 —
음료(생수/커피·차 등)는 용량이 핵심 스펙이지만, 정육·과자·조미료 같은 나머지
식품은 용량이라는 축 자체가 무의미하다. 그래서 category가 '식품'일 때만
is_beverage 여부를 함께 판별해 용량 축을 한 단계 더 좁힌다(수량은 음료가 아닌
식품에도 여전히 의미 있는 축이라 그대로 둔다)."""

from openai import AsyncOpenAI
from pydantic import BaseModel

from .agents.base import format_results_block, parse_json_object
from .config import settings
from .schemas import SearchResult

CATEGORIES: list[str] = [
    "패션의류/잡화",
    "뷰티",
    "출산/유아동",
    "식품",
    "주방용품",
    "생활용품",
    "홈인테리어",
    "가전디지털",
    "스포츠/레저",
    "자동차용품",
    "도서/음반/DVD",
    "완구/취미",
    "문구/오피스",
    "헬스/건강식품",
    "반려동물용품",
    "국내여행/해외여행",
]

# 용량(mL/L/g/kg) 스펙이 실제 구매 기준으로 쓰이는 카테고리 — 스킨케어/향수,
# 분유/물티슈, 세제/샴푸, 사료, 오일/워셔액처럼 액체·중량이 상품 정체성의
# 일부인 경우만 넣는다. 패션·가전·도서처럼 mL/L/kg 스펙 자체가 없는 카테고리는
# 빠진다.
VOLUME_RELEVANT_CATEGORIES: set[str] = {
    "뷰티",
    "출산/유아동",
    "식품",
    "주방용품",
    "생활용품",
    "헬스/건강식품",
    "반려동물용품",
    "자동차용품",
}

# 수량(묶음/세트 개수) 축은 용량보다 훨씬 넓게 걸린다 — intent.py의 개수 단위
# (개/병/팩/박스/세트/캔/봉지/포/장/권/벌/족/대)가 도서('권')·패션('벌'·'족')·
# 가전('대')처럼 용량 개념이 없는 카테고리까지 가로지르기 때문이다. 이 묶음
# 단위 자체가 무의미한 카테고리는 국내/해외여행(숙박·인원 단위라 이 4축과 무관)
# 하나뿐이라 그것만 뺀다.
QUANTITY_RELEVANT_CATEGORIES: set[str] = set(CATEGORIES) - {"국내여행/해외여행"}

# 위 두 집합에 속해도, '식품'은 그 안에서 음료가 아니면 용량 축을 추가로 뺀다.
VOLUME_REQUIRES_BEVERAGE_CHECK: set[str] = {"식품"}

_CATEGORY_LIST_TEXT = "\n".join(f"- {c}" for c in CATEGORIES)

CLASSIFY_INSTRUCTIONS = (
    "당신은 쇼핑 검색어를 아래 카테고리 목록 중 하나로 분류하는 에이전트입니다. "
    "질의와 검색 결과를 참고해 가장 알맞은 카테고리 하나만 고르세요. "
    "목록에 없는 카테고리를 새로 만들지 말고, 애매하더라도 목록 중 가장 가까운 "
    "것 하나를 고르세요. "
    "category가 '식품'이면, 그 상품이 마시는 음료(생수/탄산음료/주스/커피/차/이온음료 등 "
    "'커피·차·음료'나 '생수·음료' 같은 음료 서브카테고리에 해당하는지)인지도 "
    "is_beverage로 함께 표시하세요 — 정육·수산물·과자·조미료·쌀/잡곡 등 마시는 음료가 "
    "아닌 식품은 false입니다. category가 '식품'이 아니면 is_beverage는 항상 false로 두세요. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{"category": "...", "is_beverage": true|false}\n\n'
    f"카테고리 목록:\n{_CATEGORY_LIST_TEXT}"
)

class CategoryClassification(BaseModel):
    category: str | None = None
    is_beverage: bool = False


def build_classify_prompt(query: str, search_results: list[SearchResult]) -> str:
    results_block = format_results_block(search_results)
    return f"{CLASSIFY_INSTRUCTIONS}\n\n사용자 질의: {query}\n\n검색 결과:\n{results_block}"


async def classify_category(query: str, search_results: list[SearchResult]) -> CategoryClassification:
    """질의를 16개 대분류 중 하나로 분류하고, '식품'이면 음료 여부도 함께
    판별한다. 실패하거나 목록에 없는 값이 나오면 category=None을 반환하고,
    호출부는 이를 '분류 불확실 → 기존처럼 전 축 유지'로 안전하게 처리한다
    (용량/수량을 잘못 숨기는 것보다 안 물어볼 걸 한 번 더 묻는 게 낫다)."""
    try:
        # max_retries=0 - 사용자 요청(2026-08-15: "너무 느려 더 빠르게"). 실패해도
        # 호출부가 이미 폴백(분류 불확실 처리)을 갖고 있어 SDK 재시도로 얻는
        # 이득보다 지연 비용이 크다.
        #
        # 토큰 절약(2026-08-19) - groq_model(120b)이 아니라 groq_refine_model(20b)을
        # 쓴다. 이 함수는 검색마다(스타일 가이드 카테고리 게이트 때문에) 무조건
        # 한 번씩 불리는데, groq_model(120b)은 이미 propose의 groq 슬롯 + judge +
        # style_guide까지 몰려있어 이 계정의 Groq 일일 한도(모델당 200,000 토큰)를
        # 가장 먼저 소진하는 모델이었다(실측 2026-08-19: 120b가 199,917/200,000
        # 소진된 채로 429). 반면 refine 전용이던 20b는 매 요청 1건뿐이라 여유가
        # 크다. 카테고리 분류(16개 중 하나 + 불리언 하나 고르기)는 반드시 120b급
        # 추론이 필요한 작업이 아니라고 판단해 옮긴다 - 예전에 propose+분류+OCR을
        # *한꺼번에* 20b로 몰았다가 20b가 오히려 더 빨리 고갈된 적이 있었지만
        # (config.py 주석 참고), 그건 세 가지를 한 번에 옮긴 결과였고 여기서는
        # 분류 하나만 옮겨 20b·120b 두 예산에 걸리는 부하를 좀 더 고르게 나눈다.
        # 실패해도 categoy=None으로 안전하게 처리되니(위 docstring), 분류 품질이
        # 이 모델에서 떨어지더라도 파이프라인이 깨지지는 않는다.
        client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_api_base, max_retries=0)
        response = await client.chat.completions.create(
            model=settings.groq_refine_model,
            messages=[{"role": "user", "content": build_classify_prompt(query, search_results)}],
        )
        data = parse_json_object(response.choices[0].message.content or "")
        category = data.get("category")
        category = category if category in CATEGORIES else None
        is_beverage = bool(data.get("is_beverage")) if category == "식품" else False
        return CategoryClassification(category=category, is_beverage=is_beverage)
    except Exception:
        return CategoryClassification()
