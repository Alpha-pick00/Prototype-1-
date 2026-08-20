import asyncio
import json
import logging

import jwt

logging.basicConfig(level=logging.INFO)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import TypeAdapter

from . import autocomplete, history, preferences
from .agents import gpt as gpt_agent
from .auth import google as google_auth
from .auth import kakao as kakao_auth
from .auth import naver as naver_auth
from .auth.session import issue_session_token, verify_session_token
from .debate import (
    check_clarify_facets,
    run_brand_price,
    run_elevenst_only_debate,
    run_elevenst_only_debate_stream,
)
from .ocr import cleanup as ocr_cleanup
from .ocr import google_vision as google_vision_ocr
from .schemas import (
    AuthResponse,
    BrandPriceResponse,
    BulkDecideResponse,
    ClarifyAskRequest,
    ClarifyAskResponse,
    ClarifyResponse,
    DecideRequest,
    DecideResponse,
    DecideResultUnion as DecideResult,
    GoogleAuthRequest,
    HistoryEntry,
    OAuthCodeRequest,
    OcrExtractResponse,
    PreferenceRecordRequest,
    SaveHistoryRequest,
    User,
)

_decide_result_adapter = TypeAdapter(DecideResult)

app = FastAPI(title="αlpha Pick Purchase Decision API")

# GitHub Pages(정적 프론트엔드)에서 이 API를 브라우저로 직접 호출하므로 CORS 허용이 필요하다.
# 인증이 없는 API라 origin을 넓게 열어도 데이터 유출 위험은 없지만, "*"로 두면 아무 사이트나
# 이 API(유료 LLM 호출)를 자기 페이지에 박아 넣고 우리 예산을 소모시킬 수 있어 알려진
# origin으로만 제한한다.
# 2026-08-18("Vercel로 배포해줘") - GitHub Pages와 별개로 Vercel에도 같은 프론트엔드를
# 배포했다. Vercel은 배포마다 고유 URL도 발급하지만(예: alpha-pick-<해시>-<팀>.vercel.app),
# 실사용자는 고정 프로덕션 별칭만 쓰므로 그 하나만 허용한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://alpha-pick00.github.io",
        "https://alpha-pick-jet.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

autocomplete.seed()

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    try:
        return verify_session_token(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="세션이 만료되었거나 유효하지 않습니다.") from exc


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User | None:
    """get_current_user와 달리 비로그인/유효하지 않은 토큰이어도 401을 던지지
    않고 None을 반환한다 - 사용자 페르소나(2026-08-15) 조회처럼 "로그인했으면
    반영하고, 아니면 그냥 세션 값만 쓴다"는 선택적 개인화에 쓴다."""
    if credentials is None:
        return None
    try:
        return verify_session_token(credentials.credentials)
    except jwt.PyJWTError:
        return None


def _autocomplete_terms(request: DecideRequest, result: DecideResult) -> list[str]:
    """검색어 + 파이프라인이 이미 만들어낸 모든 상품/브랜드 후보를 자동완성 인덱스에 반영한다.

    judge가 최종 선택한 하나만 남기면, 각 에이전트가 실제 검색 결과에서 찾아낸
    나머지 후보와 clarify 단계에서 뽑힌 브랜드/용량/수량은 그냥 버려진다.
    검색 1건당 이미 검증된 상품 단어가 여러 개 나오므로 전부 모은다.
    """
    terms = [request.query]

    if isinstance(result, DecideResponse):
        terms.append(result.decision.product_name)
        terms.extend(p.product_name for p in result.proposals if p.error is None)

    elif isinstance(result, BulkDecideResponse):
        for option in result.decision.options:
            terms.append(option.brand)
            terms.append(option.product_name)
        for proposal in result.proposals:
            if proposal.error is not None:
                continue
            for option in proposal.options:
                terms.append(option.brand)
                terms.append(option.product_name)

    elif isinstance(result, ClarifyResponse):
        terms.extend(result.options.brands)
        terms.extend(result.options.volumes)
        terms.extend(result.options.quantities)

    elif isinstance(result, BrandPriceResponse) and result.option:
        terms.append(result.option.product_name)

    return terms


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/autocomplete", response_model=list[str])
async def get_autocomplete(q: str, limit: int = 8) -> list[str]:
    return await autocomplete.suggest_merged(q, limit)


@app.post("/clarify/ask", response_model=ClarifyAskResponse)
async def clarify_ask(request: ClarifyAskRequest) -> ClarifyAskResponse:
    """이번 라운드에 물어볼 축(브랜드/제품/용량/개수)의 후보들을 실제 상담원처럼
    자연스러운 질문 문장으로 바꾼다 — 프론트가 "브랜드를 선택하면 좁혀드려요"
    같은 고정 라벨 대신 이 문장을 채팅 말풍선으로 먼저 보여준다."""
    message = await gpt_agent.generate_clarify_question(request.query, request.options)
    return ClarifyAskResponse(message=message)


@app.get("/auth/me", response_model=User)
def auth_me(user: User = Depends(get_current_user)) -> User:
    return user


@app.get("/history", response_model=list[HistoryEntry])
def get_history(user: User = Depends(get_current_user)) -> list[HistoryEntry]:
    return history.list_entries(user)


@app.post("/history", response_model=HistoryEntry)
def save_history(
    request: SaveHistoryRequest, user: User = Depends(get_current_user)
) -> HistoryEntry:
    return history.add_entry(user, request.query, request.result)


@app.delete("/history/{entry_id}")
def delete_history_entry(entry_id: str, user: User = Depends(get_current_user)) -> dict[str, str]:
    history.delete_entry(user, entry_id)
    return {"status": "ok"}


@app.delete("/history")
def delete_all_history(user: User = Depends(get_current_user)) -> dict[str, str]:
    history.clear_entries(user)
    return {"status": "ok"}


@app.post("/preferences")
def record_preference(
    request: PreferenceRecordRequest, user: User = Depends(get_current_user)
) -> dict[str, str]:
    """사용자 페르소나(2026-08-15) - 로그인한 사용자가 clarify에서 facet/브랜드
    값을 하나 고를 때마다 프론트가 fire-and-forget으로 호출한다. 계정에 누적된
    선호도는 이후 검색의 /decide/clarify가 facet 옵션 순서에 소프트하게
    반영한다(app.preferences.get_top_preferences, app.debate._apply_persona_ordering)."""
    preferences.record(user, request.label, request.value)
    return {"status": "ok"}


@app.post("/auth/google", response_model=AuthResponse)
async def auth_google(request: GoogleAuthRequest) -> AuthResponse:
    try:
        user = await google_auth.fetch_user(request.access_token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"구글 로그인에 실패했습니다: {exc}") from exc
    return AuthResponse(token=issue_session_token(user), user=user)


@app.post("/auth/kakao", response_model=AuthResponse)
async def auth_kakao(request: OAuthCodeRequest) -> AuthResponse:
    try:
        user = await kakao_auth.exchange_code(request.code, request.redirect_uri)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"카카오 로그인에 실패했습니다: {exc}") from exc
    return AuthResponse(token=issue_session_token(user), user=user)


@app.post("/auth/naver", response_model=AuthResponse)
async def auth_naver(request: OAuthCodeRequest) -> AuthResponse:
    try:
        user = await naver_auth.exchange_code(request.code, request.state)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"네이버 로그인에 실패했습니다: {exc}") from exc
    return AuthResponse(token=issue_session_token(user), user=user)


@app.post("/ocr/extract", response_model=OcrExtractResponse)
async def ocr_extract(image: UploadFile) -> OcrExtractResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")

    ocr_result = await google_vision_ocr.extract_text(image_bytes)
    cleaned = await ocr_cleanup.clean(ocr_result.text) if not ocr_result.error else None
    return OcrExtractResponse(ocr=ocr_result, cleaned=cleaned)


@app.post("/decide", response_model=DecideResult)
async def decide(request: DecideRequest, background_tasks: BackgroundTasks) -> DecideResult:
    try:
        if request.brand:
            result = await run_brand_price(request.query, request.brand)
        else:
            # 메인 검색 흐름은 11번가 오픈 API 전용 경로(run_elevenst_only_debate)를
            # 쓴다. skip_intent_check(재질문 스킵) 구분은 무의미해 두 분기를
            # 하나로 합쳤다 - 이 경로엔 애초에 되묻기(clarify)가 없다. base_query가
            # 있으면(AI 상세검색 드릴다운 후속 턴) 재검색 대신 구조적 필터링으로
            # 좁힌다(_search_candidates 참고).
            result = await run_elevenst_only_debate(request.query, base_query=request.base_query)
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: 제안 전부 실패, ValueError: judge 응답에서 JSON을 못 찾음
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        # 외부 LLM API 오류 등 예상 못한 실패는 내부 정보를 노출하지 않고 502로 감싼다.
        raise HTTPException(
            status_code=502, detail="구매 결정을 처리하는 중 오류가 발생했습니다."
        ) from exc

    background_tasks.add_task(autocomplete.record_terms, _autocomplete_terms(request, result))
    return result


@app.post("/decide/stream")
async def decide_stream(request: DecideRequest) -> StreamingResponse:
    """/decide와 같은 일을 하지만, 검색 완료·에이전트별 제안 완료·심사 단계마다
    한 줄씩(NDJSON) 흘려보낸다. 그래야 프론트가 세 에이전트를 다 기다리지 않고
    먼저 끝난 제안부터 화면에 보여줄 수 있다. 응답 헤더가 이미 200으로 나간
    뒤라 실패해도 HTTP 상태 코드를 바꿀 수 없으므로, 에러도 "error" 이벤트로
    흘려보낸다 — 프론트는 이 타입을 보고 에러 처리한다."""

    async def event_generator():
        try:
            if request.brand:
                result: DecideResult = await run_brand_price(request.query, request.brand)
                yield json.dumps({"type": "final", "result": result.model_dump()}) + "\n"
            else:
                result = None
                # 메인 검색 흐름은 decide()와 같은 이유로 run_elevenst_only_debate_stream을
                # 쓴다(위 decide() 주석 참고 - clarify 개념이 없어 skip_intent_check
                # 분기도 함께 없앴다).
                async for event in run_elevenst_only_debate_stream(
                    request.query, base_query=request.base_query
                ):
                    if event["type"] == "final":
                        result = _decide_result_adapter.validate_python(event["result"])
                    yield json.dumps(event) + "\n"
        except (RuntimeError, ValueError) as exc:
            # RuntimeError: 제안 전부 실패, ValueError: judge 응답에서 JSON을 못 찾음
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return
        except Exception:
            # 외부 LLM API 오류 등 예상 못한 실패는 내부 정보를 노출하지 않고 감싼다.
            yield json.dumps(
                {"type": "error", "message": "구매 결정을 처리하는 중 오류가 발생했습니다."}
            ) + "\n"
            return

        if result is not None:
            asyncio.create_task(
                asyncio.to_thread(autocomplete.record_terms, _autocomplete_terms(request, result))
            )

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/decide/clarify", response_model=ClarifyResponse)
async def decide_clarify(
    request: DecideRequest, user: User | None = Depends(get_optional_user)
) -> ClarifyResponse:
    """AI 상세검색(2026-08-12) - "음료수"처럼 짧고 애매한 검색어를 11번가 실제
    검색 상품명에 근거해 DeepSeek이 몇 가지 기준(카테고리/브랜드/용량 등)으로 좁혀나가게
    제안한다. 프론트는 짧은 검색어에 한해 elevenst-only 빠른 경로를 타기 전에 이걸
    먼저 불러보고, options.facets가 비어 있으면(=명확한 검색어) 그대로 원래
    빠른 경로로 넘어간다 - 이 엔드포인트 자체가 대부분의 검색어에 대해 검색/LLM
    호출 없이 즉시 빈 결과로 끝나므로(check_clarify_facets 참고) 매 검색마다
    호출해도 비용이 거의 없다.

    사용자 페르소나(2026-08-15) - 로그인했으면 계정에 영구 누적된 선호도
    (app.preferences)를 먼저 깔고, 이번 세션에서 프론트가 들고 있다가 보낸
    session_preferences로 덮어써 최신 선택을 우선한다."""
    persona: dict[str, str] = {}
    if user is not None:
        persona.update(preferences.get_top_preferences(user))
    if request.session_preferences:
        persona.update(request.session_preferences)
    return await check_clarify_facets(request.query, base_query=request.base_query, persona=persona)


@app.post("/decide/elevenst-only", response_model=DecideResponse)
async def decide_elevenst_only(request: DecideRequest) -> DecideResponse:
    """/decide와 실질적으로 같은 경로(run_elevenst_only_debate)를 별도
    엔드포인트로도 노출해둔다(로컬 실험/검증 전용, 프론트엔드는 쓰지 않음)."""
    try:
        return await run_elevenst_only_debate(request.query, base_query=request.base_query)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="11번가 전용 처리 중 오류가 발생했습니다."
        ) from exc
