import logging
import re
from collections import defaultdict

from openai import AsyncOpenAI

from ..config import settings
from ..schemas import BulkProposal, ChallengeVerdict, ClarifyFacet, SearchResult
from .base import (
    build_bulk_prompt,
    build_challenge_prompt,
    build_facet_clarify_prompt,
    build_facet_clarify_prompt_for_labels,
    filter_bulk_options,
    parse_json_array,
    parse_json_object,
)

logger = logging.getLogger(__name__)

# DeepSeek은 OpenAI 호환 API라 openai SDK를 base_url만 바꿔서 그대로 쓴다.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _client() -> AsyncOpenAI:
    # max_retries=0 - 사용자 요청(2026-08-15: "너무
    # 느려 더 빠르게"). 실패해도 호출부가 이미 폴백을 갖고 있어 SDK 재시도로
    # 얻는 이득보다 지연 비용이 크다.
    return AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=DEEPSEEK_BASE_URL, max_retries=0)


async def challenge_candidates(
    query: str,
    candidates: list[dict],
    search_results: list[SearchResult],
    candidate_pages: dict[str, str] | None = None,
    coupang_results: list[SearchResult] | None = None,
    naver_results: list[SearchResult] | None = None,
) -> list[ChallengeVerdict]:
    """ADK 파이프라인의 정상 challenge 단계(LlmAgent + LiteLlm)와 별개로,
    파이프라인 세션 밖에서(2026-08-16, relaxed fallback 하드닝) 같은 그라운딩
    검증을 태울 때 쓰는 단발 호출 버전 - build_challenge_prompt를 그대로
    재사용해 프롬프트/기준이 정상 경로와 어긋나지 않는다. 실패(API 오류·파싱
    실패)하면 빈 리스트를 반환한다 - 호출부는 이를 "검증 안 됨"으로 처리하고,
    이미 찾은 유일한 후보를 검증 인프라 장애 때문에 버리지 않는다."""
    try:
        client = _client()
        prompt = build_challenge_prompt(
            query, candidates, search_results, candidate_pages, coupang_results, naver_results
        )
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return [ChallengeVerdict(**v) for v in parse_json_array(response.choices[0].message.content or "")]
    except Exception:
        logger.warning("relaxed fallback 후보 challenge 검증 실패 — 미검증으로 진행", exc_info=True)
        return []


async def propose_bulk(query: str, search_results: list[SearchResult]) -> BulkProposal:
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": build_bulk_prompt(query, search_results)}],
        )
        options = parse_json_array(response.choices[0].message.content or "")
        options = filter_bulk_options(options, search_results)
        return BulkProposal(agent="deepseek", options=options)
    except Exception as exc:
        return BulkProposal(agent="deepseek", error=str(exc))


MAX_FACETS = 4
MAX_OPTIONS_PER_FACET = 6
# "브랜드"/"제조사" 기준은 사용자 요청(2026-08-12: "브랜드가 2,3개 정도만 뜨는데
# ... 찾기 기능도 있었으면")으로 다른 기준보다 훨씬 넓게 보여준다 - 프론트에
# 검색으로 걸러볼 수 있는 입력창을 붙였으니(SearchResults.tsx) 잘라내기 상한을
# 낮게 둘 이유가 없다.
MAX_BRAND_OPTIONS = 15
# 순수 브랜드/제조사만 가리킨다 - _drop_brand_facet_when_device_model_present가
# 이 패턴으로 "브랜드 facet만" 골라 지우므로, "기종"류까지 여기 섞으면 그쪽도
# 같이 지워진다(의도한 동작 아님). 상한 넓히기 용도는 아래 _WIDE_CAP_LABEL_PATTERN.
_BRAND_LABEL_PATTERN = re.compile(r"브랜드|제조사")
# "기종"류(호환기종/핸드폰 기종)도 브랜드처럼 넓게 보여준다(사용자 리포트,
# 2026-08-14: "핸드폰 케이스" 검색에서 호환 기기 종류가 많아 6개로는 자기
# 기종이 안 보일 수 있음 - 다나와 실제 화면도 "+" 펼치기로 174개까지 보여준다 -
# 실사용자가 정확히 하나를 아는 축이라 검색으로 거를 수 있으면 상한을 낮게 둘
# 이유가 없다).
_WIDE_CAP_LABEL_PATTERN = re.compile(r"브랜드|제조사|호환\s*기종|핸드폰\s*기종")
_CONTAINER_FORM_LABEL_PATTERN = re.compile(r"용기\s*형태")
# "용기형태"는 페트/캔/유리병 같은 물리적 용기 형태만 가리켜야 하는데, LLM이
# 상품명 속 구매유형 수식어를 용기형태로 잘못 묶어 넣는 경우가 있었다(사용자
# 리포트, 2026-08-14: 음료 검색에서 용기형태 선택지로 "업소용"이 나옴 - 페트/캔이
# 나와야 정상). 프롬프트를 명확히 했지만(build_facet_clarify_prompt), 코드에서도
# 한 번 더 걸러낸다.
_NON_CONTAINER_FORM_TERMS = {"업소용", "가정용", "업소", "가정", "벌크", "낱개", "묶음", "세트", "단품"}

_PURCHASE_TYPE_LABEL_PATTERN = re.compile(r"구매\s*유형")
# 같은 버그 패턴(사용자 리포트, 2026-08-14: "핸드폰 케이스" 검색에서 구매유형으로
# "해외"/"중고"가 뜸 - 상품명에 그런 단어가 없는데도 LLM이 스마트폰 시장 통념을
# 끌어와 만들어낸 값). "구매유형"은 개방형 자유 텍스트가 아니라 정품 상태/유통
# 경로를 가리키는 한정된 어휘라, 용기형태(블랙리스트)와 반대로 화이트리스트로
# 걸러낸다 - 모델명·색상·호환 기종 같은 다른 기준 값이 새어 들어와도 걸러진다.
_PURCHASE_TYPE_ALLOWED_TERMS = {
    "정품", "새제품", "미개봉", "리퍼", "리퍼비시", "중고", "전시품", "전시상품",
    "병행수입", "해외구매", "해외직구", "직구", "구매대행", "정식수입", "국내정식수입",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _dedupe_across_facets(facets: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """모든 기준(facet)을 통틀어 값 중복을 정리한다 - 같은 값이 서로 다른
    기준에 동시에 뜨거나(사용자 리포트, 2026-08-14: "핸드폰 케이스" 검색에서
    "맥세이프"가 "시리즈"에도 "특징"에도 동시에 뜸), 한 값이 다른 값 문자열에
    완전히 포함되는 부분 문자열이면(같은 리포트: "특징"에 "갤럭시S25 울트라"와
    별개로 "울트라"만 단독으로도 뜸) 더 짧거나 덜 구체적인 쪽을 버린다. 어느
    쪽이 먼저 나왔든 상관없이 더 길고 구체적인 값이 이긴다 - 프롬프트로도
    지시했지만(build_facet_clarify_prompt) LLM이 어길 수 있어 코드에서 한 번
    더 걸러낸다.

    단, '핸드폰 기종'(기기 모델) 값끼리는 부분 문자열이어도 서로 다른 값을
    흡수하지 않는다 - "아이폰17"과 "아이폰17 프로"처럼 한쪽이 다른 쪽 문자열을
    포함해도 실제로는 서로 다른 모델이라, "울트라"(그 자체로는 독립적인 값이
    아닌 조각) 같은 진짜 중복 사례와 다르다."""
    kept: dict[str, tuple[str, str]] = {}  # normalized -> (원본 값, 라벨)
    for label, options in facets:
        allow_substring_merge = label != _DEVICE_MODEL_LABEL
        for value in options:
            key = _normalize(value)
            if key in kept:
                continue
            if allow_substring_merge and any(key in existing for existing in kept):
                continue  # 이미 채택된 더 긴 값의 부분 문자열
            if allow_substring_merge:
                superseded = [existing for existing in kept if existing in key]
                for existing in superseded:
                    del kept[existing]
            kept[key] = (value, label)

    by_label: dict[str, list[str]] = {}
    for value, label in kept.values():
        by_label.setdefault(label, []).append(value)

    result: list[tuple[str, list[str]]] = []
    for label, options in facets:
        survivors = [o for o in options if o in by_label.get(label, [])]
        if survivors:
            result.append((label, survivors))
    return result


# 사용자 요청(2026-08-14: "갤럭시 전용 이렇게 없애고, 핸드폰 기종별로 선택할
# 수 있게") - 기기 브랜드별로 필터 행을 나누지 않고, 갤럭시/아이폰/아이패드를
# 통틀어 '핸드폰 기종' 기준 하나로 합친다.
_DEVICE_MODEL_LABEL = "핸드폰 기종"
# 절대다수 매물이 이 계열이라, 실제 시장을 커버하는 데는 이걸로 충분하다 -
# 기기 브랜드가 사실상 무한하니 완전한 목록은 아니다.
_DEVICE_MODEL_PREFIXES = ("갤럭시", "아이폰", "아이패드")


def _looks_like_device_model(value: str) -> bool:
    """값이 특정 기기 모델(갤럭시/아이폰/아이패드 계열)을 가리키는지 판정한다.
    케이스·필름·거치대 같은 액세서리 검색에서는 이런 값이 상품 자체의 시리즈가
    아니라 "이 상품이 맞는 기기"를 뜻하는데, LLM이 매 호출마다 이런 값을
    카테고리/시리즈/특징 등 서로 다른 라벨에 제각각 담아 보낸다(사용자 리포트,
    2026-08-14: "핸드폰 케이스" 검색) - 실제로 어떤 라벨에 담겨 왔든 코드에서
    '핸드폰 기종' 라벨 하나로 강제 통일한다."""
    stripped = value.strip()
    return stripped.startswith(_DEVICE_MODEL_PREFIXES) and any(c.isdigit() for c in stripped)


# 사용자 요청(2026-08-14: "2020년 이후 모델로만 보이게 하는 방법 없어? 아이폰
# 12부터라던지") - 아이폰/갤럭시S/갤럭시노트는 세대 번호가 출시 연도와 거의
# 그대로 대응한다(아이폰12=2020년, 갤럭시S20=2020년, 갤럭시노트20=2020년)는
# 걸 이용해 안전하게 컷오프를 건다. 갤럭시Z(폴드/플립 - 번호 없는 1세대만
# 2019년이고 번호가 붙으면 이미 2020년 이후라 별도 컷오프가 필요 없음) ·
# 갤럭시A(넘버링이 연식과 느슨하게만 대응) · 아이패드는 신뢰할 만한 규칙이
# 없어 컷오프를 걸지 않는다 - 걸러야 할 값을 놓치는 것보다 실제로 최신인
# 값을 잘못 지우는 쪽이 더 나쁘다고 보고 보수적으로 접근한다.
_IPHONE_GEN_PATTERN = re.compile(r"^아이폰\s*(\d+)")
_GALAXY_NOTE_GEN_PATTERN = re.compile(r"^갤럭시\s*노트\s*(\d+)")
_GALAXY_S_GEN_PATTERN = re.compile(r"^갤럭시\s*S\s*(\d+)", re.IGNORECASE)
_MIN_IPHONE_GEN = 12
_MIN_GALAXY_NOTE_GEN = 20
_MIN_GALAXY_S_GEN = 20


def _is_pre_2020_device_model(value: str) -> bool:
    stripped = value.strip()
    for pattern, min_gen in (
        (_IPHONE_GEN_PATTERN, _MIN_IPHONE_GEN),
        (_GALAXY_NOTE_GEN_PATTERN, _MIN_GALAXY_NOTE_GEN),
        (_GALAXY_S_GEN_PATTERN, _MIN_GALAXY_S_GEN),
    ):
        match = pattern.match(stripped)
        if match:
            return int(match.group(1)) < min_gen
    return False


def _consolidate_device_model_values(
    facets: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    device_bucket: list[str] = []
    other: list[tuple[str, list[str]]] = []
    for label, options in facets:
        if label == _DEVICE_MODEL_LABEL:
            device_bucket.extend(o for o in options if not _is_pre_2020_device_model(o))
            continue
        kept = [o for o in options if not _looks_like_device_model(o)]
        device_bucket.extend(
            o for o in options if _looks_like_device_model(o) and not _is_pre_2020_device_model(o)
        )
        if kept:
            other.append((label, kept))
    if not device_bucket:
        return other
    return [(_DEVICE_MODEL_LABEL, device_bucket)] + other


def _merge_same_label_facets(facets: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    merged: dict[str, list[str]] = {}
    order: list[str] = []
    for label, options in facets:
        if label not in merged:
            merged[label] = []
            order.append(label)
        merged[label].extend(options)
    return [(label, merged[label]) for label in order]


def _sort_by_popularity(options: list[str], product_names: list[str]) -> list[str]:
    """LLM이 알려준 순서를 믿지 않고, 실제 상품명 목록에서 각 값이 몇 번
    등장하는지(부분 문자열 포함) 직접 세어 내림차순으로 다시 정렬한다 -
    "인기순 정렬" 요청(2026-08-12)의 실제 근거가 LLM의 자기 진술이 아니라
    검색 결과 자체가 되도록. 동률(count 같음)이면 원래 순서를 유지한다
    (sort는 stable)."""
    normalized_names = [_normalize(n) for n in product_names]
    counts = {
        option: sum(1 for name in normalized_names if _normalize(option) in name)
        for option in options
    }
    return sorted(options, key=lambda o: counts[o], reverse=True)


def _device_model_ecosystem(value: str) -> str:
    """'핸드폰 기종' 값을 기기 브랜드 계열로 묶는 내부 그룹 키(사용자에게
    보이는 라벨이 아니라 _balance_device_models_by_ecosystem의 정렬용)."""
    stripped = value.strip()
    if stripped.startswith("갤럭시"):
        return "갤럭시"
    if stripped.startswith(("아이폰", "아이패드")):
        return "아이폰"
    return "기타"


def _balance_device_models_by_ecosystem(options: list[str], product_names: list[str], cap: int) -> list[str]:
    """'핸드폰 기종'을 표본 전체 인기순으로 한 줄 정렬 후 자르면, 그 검색어에
    매물이 절대적으로 많은 브랜드(예: 갤럭시)가 상한을 다 차지해 다른 브랜드
    모델(예: 아이폰14)이 아예 안 보일 수 있다(사용자 리포트, 2026-08-14: "선택지에
    너무 갤럭시만 모여서 보여주는 경향이 있어" - "브랜드가 2,3개 정도만 뜨는데"
    (2026-08-12)와 같은 근본 원인을 '기종' 축에서 다시 겪은 것). 브랜드 facet의
    다수결 쏠림을 브랜드별 재추출(_enrich_facets_per_brand)로 풀었던 것과 같은
    발상으로, 계열별로 그룹을 나눠 각자 인기순 정렬한 뒤 라운드로빈으로 섞는다 -
    표본에 등장한 계열이면 그 계열의 1등 모델이 최소한 초반 순번을 받는다."""
    groups: dict[str, list[str]] = defaultdict(list)
    for value in options:
        groups[_device_model_ecosystem(value)].append(value)
    for key, values in groups.items():
        groups[key] = _sort_by_popularity(values, product_names)

    # 계열 순서 자체도 표본에서의 전체 등장 빈도 내림차순으로 - 그래야 라운드로빈
    # 첫 바퀴가 "가장 흔한 계열 1등 -> 그다음 흔한 계열 1등 -> ..." 순이 된다.
    ordered_ecosystems = sorted(
        groups.keys(),
        key=lambda k: sum(1 for name in product_names if _normalize(k) in _normalize(name)) if k != "기타" else -1,
        reverse=True,
    )

    result: list[str] = []
    round_idx = 0
    while len(result) < cap and any(groups[k] for k in ordered_ecosystems):
        eco = ordered_ecosystems[round_idx % len(ordered_ecosystems)]
        if groups[eco]:
            result.append(groups[eco].pop(0))
        round_idx += 1
    return result


async def extract_facets_from_names(
    query: str, product_names: list[str], required_labels: list[str] | None = None
) -> list[ClarifyFacet]:
    """다나와 검색 결과 상품명 목록만 보고, 검색어를 좁혀나갈 수 있는 기준(facet)을
    뽑아낸다(AI 상세검색, 2026-08-12 - 원래 Qwen으로 붙였다가 Model Studio 계정
    쪽 과금 플랜 활성화 문제로 이미 키가 있고 바로 되는 DeepSeek로 옮겼다).
    실패하거나(API 오류, JSON 파싱 실패 등) 아무 기준도 못 찾으면 조용히 빈
    리스트를 반환한다 - 호출자(app.debate.check_clarify_facets)가 "상세검색이
    필요 없다"와 동일하게 취급해 그대로 원래 검색 경로로 넘어간다.

    required_labels(2026-08-13, app.debate._enrich_facets_per_brand 전용) - 주어지면
    라벨을 자유롭게 고르게 두지 않고 정확히 이 라벨들만 쓰라고 프롬프트로 강제한다.
    브랜드별로 상품명을 좁혀 다시 부를 때, 매 호출마다 같은 개념을 "시리즈"/"모델"처럼
    다르게 이름 붙이면 나중에 라벨로 병합할 수 없어서다 - 그래도 모델이 지시를 어기고
    다른 라벨을 낼 수 있으니, 응답에서도 required_labels에 없는 라벨은 걸러낸다."""
    if not product_names:
        return []
    try:
        client = _client()
        prompt = (
            build_facet_clarify_prompt_for_labels(query, product_names, required_labels)
            if required_labels
            else build_facet_clarify_prompt(query, product_names)
        )
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
        )
        data = parse_json_object(response.choices[0].message.content or "")
        allowed_labels = set(required_labels) if required_labels else None
        cleaned_by_label: list[tuple[str, list[str]]] = []
        for label, options in (data.get("facets") or {}).items():
            if allowed_labels is not None and str(label) not in allowed_labels:
                continue
            if not isinstance(options, list):
                continue
            cleaned = [str(o).strip() for o in options if str(o).strip()]
            if _CONTAINER_FORM_LABEL_PATTERN.search(str(label)):
                cleaned = [c for c in cleaned if c not in _NON_CONTAINER_FORM_TERMS]
            if _PURCHASE_TYPE_LABEL_PATTERN.search(str(label)):
                cleaned = [c for c in cleaned if _normalize(c) in _PURCHASE_TYPE_ALLOWED_TERMS]
            if cleaned:
                cleaned_by_label.append((str(label), cleaned))

        # 갤럭시/아이폰/아이패드 모델명이 어떤 라벨에 담겨 왔든 '핸드폰 기종'
        # 하나로 모으고, 같은 라벨로 다시 합친 뒤, 남은 중복·부분 문자열
        # 값을 기준을 통틀어 한 번 더 걸러낸다.
        cleaned_by_label = _consolidate_device_model_values(cleaned_by_label)
        cleaned_by_label = _merge_same_label_facets(cleaned_by_label)
        cleaned_by_label = _dedupe_across_facets(cleaned_by_label)

        facets: list[ClarifyFacet] = []
        for label, cleaned in cleaned_by_label:
            if not cleaned:
                continue
            if allowed_labels is None and len(set(cleaned)) < 2:
                # 값이 하나뿐인 기준(예: "핸드폰" 검색에 "카테고리: 스마트폰")은
                # 골라도 아무것도 안 좁혀지니 물어볼 이유가 없다(사용자 요청,
                # 2026-08-13: "카테고리에 스마트폰은 있으면 안되고"). required_labels가
                # 있는 브랜드별 재추출에서는 적용 안 한다 - 그 브랜드가 그 기준에서
                # 값이 하나뿐이어도(예: APPLE 시리즈가 1개), 다른 브랜드 값과 합쳐질
                # 옵션이라 여전히 쓸모 있다.
                continue
            cap = MAX_BRAND_OPTIONS if _WIDE_CAP_LABEL_PATTERN.search(label) else MAX_OPTIONS_PER_FACET
            if label == _DEVICE_MODEL_LABEL:
                cleaned = _balance_device_models_by_ecosystem(cleaned, product_names, cap)
            else:
                cleaned = _sort_by_popularity(cleaned, product_names)[:cap]
            facets.append(ClarifyFacet(label=label, options=cleaned))
        return facets[:MAX_FACETS]
    except Exception:
        return []
