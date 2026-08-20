import asyncio
import re

from openai import AsyncOpenAI

from ..agents.base import parse_json_object
from ..config import settings
from ..schemas import OcrCleanupResult

CLEANUP_INSTRUCTIONS = (
    "당신은 OCR로 추출한 텍스트를 정리하는 도우미입니다. OCR 특성상 줄바꿈이 뒤섞이거나, "
    "같은 내용이 중복되거나, 순서가 뒤틀려 있을 수 있습니다. "
    "아래 OCR 원본 텍스트를 읽기 쉽게 정리하세요. "
    "이미지를 직접 보는 게 아니므로 원본에 없는 내용을 새로 지어내지 말고, "
    "원본에 있는 내용만 재배열하고 중복을 제거하세요. "
    "추가로 이 텍스트가 상품 포장/라벨/영수증이라면, 쇼핑 검색엔진에 그대로 넣었을 때 "
    "가장 정확한 결과가 나올 검색어를 만드세요 — 브랜드명과 상품명(용량·모델명 등 "
    "식별에 필요한 스펙 포함)만 남기고, 가격/바코드/영양정보/광고 문구/중복 표기는 "
    "전부 제외하세요. 상품을 특정할 수 없으면 search_query를 빈 문자열로 두세요. "
    "반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트를 덧붙이지 마세요.\n\n"
    '{"cleaned_text": "정리된 텍스트", '
    '"search_query": "쇼핑 검색에 바로 쓸 짧은 검색어(브랜드+상품명, 불필요한 정보 제외)", '
    '"notes": "정리하면서 처리한 내용(중복 제거, 줄바꿈 정리 등). 없으면 빈 문자열"}'
)


# 상품 라벨 OCR 원본에 거의 항상 섞여 나오는 잡음 줄(영양정보/바코드/제조원
# 같은) 패턴 - Groq 정제가 실패했을 때 쓰는 규칙 기반 최후 수단 필터라
# 완벽할 필요는 없고, 검색어로 쓰기에 명백히 방해되는 줄만 걸러내면 된다.
_NOISE_LINE_PATTERNS = [
    re.compile(r"^[\d\s.,%()~-]+$"),  # 순수 숫자/기호로만 된 줄
    re.compile(r"kcal|1일\s*영양성분|영양정보|영양성분"),
    re.compile(r"나트륨|지방|콜레스테롤|탄수화물|단백질|당류|포화지방|트랜스지방"),
    re.compile(r"^\d{8,}$"),  # 바코드
    re.compile(r"제조원|판매원|유통기한|제조일자|원재료명|보관방법|반품|교환|고객센터|080-|1588-|1577-"),
]


def _fallback_local_cleanup(raw_text: str) -> str:
    """Groq 정제가 (재시도 후에도) 실패했을 때 쓰는 규칙 기반 대체 정제 -
    LLM 호출 없이 항상 즉시 동작한다는 게 핵심이라, 완벽한 정제보다는 검색어를
    노골적으로 망치는 줄(영양정보/바코드/제조원 등)만 제거하는 최소한의
    필터다(사용자 요청, 2026-08-14: "정제를 안하고 모든 텍스트를 다 보내는
    경우" - Groq 호출이 실패하면 프론트가 정제 안 된 원본 OCR 텍스트를 통째로
    검색어로 써버려서 검색 품질이 떨어졌다). 앞쪽 몇 줄만 남기는 이유 - 상품
    라벨은 보통 브랜드/상품명이 가장 먼저(가장 크게) 나오고, 뒤로 갈수록
    성분표·법적 고지 같은 잡음이 많아진다."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        if any(pattern.search(line) for pattern in _NOISE_LINE_PATTERNS):
            continue
        kept.append(line)
        if len(kept) >= 3:
            break
    return " ".join(kept) if kept else raw_text.strip()[:100]


async def _call_groq_cleanup(raw_text: str) -> OcrCleanupResult:
    # max_retries=0 - 사용자 요청(2026-08-15: "너무
    # 느려 더 빠르게"). SDK 자체 재시도(지수 백오프)와 아래 clean()의 수동
    # 재시도(1초 고정 지연, _MAX_ATTEMPTS)가 겹치면 실패 시 지연이 배가되므로,
    # 재시도는 clean() 쪽 한 곳에서만 통제한다.
    client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_api_base, max_retries=0)
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": f"{CLEANUP_INSTRUCTIONS}\n\nOCR 원본 텍스트:\n{raw_text}"}],
    )
    data = parse_json_object(response.choices[0].message.content or "")
    return OcrCleanupResult(**data)


# rate limit/네트워크 일시 오류처럼 재시도하면 성공할 만한 실패를 한 번 더
# 봐준다(사용자 리포트, 2026-08-14) - 그래도 실패하면 _fallback_local_cleanup으로
# 넘어간다.
_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 1.0


async def clean(raw_text: str) -> OcrCleanupResult:
    if not raw_text.strip():
        return OcrCleanupResult(error="정리할 OCR 텍스트가 없습니다.")

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return await _call_groq_cleanup(raw_text)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_DELAY_SECONDS)

    return OcrCleanupResult(cleaned_text=_fallback_local_cleanup(raw_text), error=str(last_error))
