from openai import AsyncOpenAI

from ..config import settings
from .base import build_clarify_ask_prompt, build_recommend_prompt, parse_json_object

# 이 모듈이 담당하는 에이전트 슬롯은 스키마/프론트엔드/테스트 전반에서
# agent="gpt"로 식별된다(파일명·함수명도 그대로) - 하지만 실제로 호출하는
# 모델은 Qwen이다. DashScope가 OpenAI SDK와 호환되는 엔드포인트를 제공해서,
# openai SDK를 base_url만 바꿔 그대로 쓴다(agents/deepseek.py와 동일한
# 패턴). agent="gpt" 식별자 자체를 "qwen"으로 바꾸지 않은 이유 - AgentName
# 리터럴, DB에 저장된 과거 기록, 프론트엔드 타입, 테스트 픽스처 등 수십
# 곳에 걸쳐 있어 그 리네임 자체가 훨씬 큰 변경이 된다. 사용자에게 보이는
# 이름만 frontend/src/app/components/SearchResults.tsx의 AGENT_LABEL에서
# "Qwen"으로 바꿔뒀다.


def _client() -> AsyncOpenAI:
    # max_retries=0 - 사용자 요청(2026-08-15: "너무
    # 느려 더 빠르게"). 실패해도 호출부가 이미 폴백을 갖고 있어 SDK 재시도로
    # 얻는 이득보다 지연 비용이 크다.
    return AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_api_base, max_retries=0)


# qwen3.7-plus는 DashScope 쪽에서 기본적으로 "thinking mode"(내부 추론 과정을
# 다 생성한 뒤에야 응답을 반환)가 켜져 있다(2026-08-20 실측 - enable_thinking을
# 안 주면 단순 질문 하나에도 20~95초가 걸린다, extra_body={"enable_thinking":
# False}를 주면 2초로 줄어든다). 이 프로젝트가 쓰는 프롬프트는 전부 JSON 한
# 덩어리만 필요해서 추론 과정 자체가 필요 없다 - 이 모듈의 모든 호출에 끈다.
_DISABLE_THINKING = {"enable_thinking": False}


_CLARIFY_ASK_FALLBACK = "몇 가지 후보를 찾았어요 — 아래에서 골라주시겠어요?"


async def generate_clarify_question(query: str, options: list[str]) -> str:
    """이번 라운드에 물어봐야 할 축(브랜드/제품/용량/개수)의 후보들을 실제
    상담원처럼 자연스러운 한 질문으로 바꾼다 — 프론트가 "브랜드를 선택하면
    좁혀드려요" 같은 고정 라벨 대신 이 문장을 채팅 말풍선으로 보여준다.
    호출 실패 시 고정 안내 문구로 대체한다."""
    if not options:
        return _CLARIFY_ASK_FALLBACK
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.qwen_model,
            messages=[{"role": "user", "content": build_clarify_ask_prompt(query, options)}],
            response_format={"type": "json_object"},
            extra_body=_DISABLE_THINKING,
        )
        data = parse_json_object(response.choices[0].message.content or "")
        return data.get("message") or _CLARIFY_ASK_FALLBACK
    except Exception:
        return _CLARIFY_ASK_FALLBACK


async def recommend_best(query: str, candidates: list[dict]) -> tuple[int, str] | None:
    """11번가 검증 후보(app.debate.run_elevenst_only_debate) 중 가장 추천할
    만한 것을 LLM이 고른다 - 가격만이 아니라 리뷰 수/구매만족도까지 본다.
    실패(키 없음·API 오류·범위 밖 index)하면 None - 호출부가 최저가 규칙
    기반으로 폴백한다."""
    if not candidates:
        return None
    try:
        client = _client()
        response = await client.chat.completions.create(
            model=settings.qwen_model,
            messages=[{"role": "user", "content": build_recommend_prompt(query, candidates)}],
            response_format={"type": "json_object"},
            extra_body=_DISABLE_THINKING,
        )
        data = parse_json_object(response.choices[0].message.content or "")
        index = int(data.get("index"))
        if not (0 <= index < len(candidates)):
            return None
        return index, str(data.get("reasoning") or "").strip()
    except Exception:
        return None
