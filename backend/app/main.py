import asyncio
import json
import logging
from contextlib import asynccontextmanager

import jwt

logging.basicConfig(level=logging.INFO)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import TypeAdapter

from . import autocomplete, danawa, decision_cache, history, popularity_scheduler, preferences
from .agents import gpt as gpt_agent
from .auth import google as google_auth
from .auth import kakao as kakao_auth
from .auth import naver as naver_auth
from .auth.session import issue_session_token, verify_session_token
from .debate import (
    check_clarify_facets,
    run_brand_price,
    run_danawa_only_debate,
    run_danawa_only_debate_stream,
    run_debate,
    run_debate_stream,
    run_single_debate,
    run_single_debate_stream,
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    popularity_scheduler.start()
    yield
    popularity_scheduler.stop()


app = FastAPI(title="αlpha Pick Purchase Decision API", lifespan=lifespan)

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


async def _resolve_danawa_urls(result: DecideResult) -> DecideResult:
    """최종 추천 URL이 다나와 가격비교 페이지면, 사용자가 실제로 구매할 수
    있는 최저가 판매처 링크로 바꿔치기한다 — 다나와 페이지 자체는 여러 판매처를
    나열만 할 뿐 바로 살 수 있는 곳이 아니다(danawa.py 참고). 다나와가 아니거나
    해석에 실패하면 원래 값 그대로 둔다. 최종 결과에만 적용하고 proposals의
    나머지 후보 URL은 그대로 둔다 — 사용자가 실제로 클릭할 하나만 바꾸면 된다."""
    if isinstance(result, DecideResponse):
        result.decision.url, result.decision.retailer = await danawa.resolve_lowest_price(
            result.decision.url, result.decision.retailer
        )
    elif isinstance(result, BulkDecideResponse):
        resolved = await asyncio.gather(
            *(danawa.resolve_lowest_price(o.url, o.retailer) for o in result.decision.options)
        )
        for option, (url, retailer) in zip(result.decision.options, resolved):
            option.url, option.retailer = url, retailer
    elif isinstance(result, BrandPriceResponse) and result.option:
        result.option.url, result.option.retailer = await danawa.resolve_lowest_price(
            result.option.url, result.option.retailer
        )
    return result


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


def _compute_persona(request: DecideRequest, user: User | None) -> dict[str, str]:
    """사용자 페르소나(2026-08-15) - 로그인했으면 계정에 영구 누적된 선호도
    (app.preferences)를 먼저 깔고, 이번 세션에서 프론트가 들고 있다가 보낸
    session_preferences로 덮어써 최신 선택을 우선한다. 원래 /decide/clarify
    전용이었는데, 그 사전 호출을 없애면서(check_clarify_facets 참고) /decide·
    /decide/stream이 페르소나 반영을 직접 넘겨받아야 한다."""
    persona: dict[str, str] = {}
    if user is not None:
        persona.update(preferences.get_top_preferences(user))
    if request.session_preferences:
        persona.update(request.session_preferences)
    return persona


@app.post("/decide", response_model=DecideResult)
async def decide(
    request: DecideRequest,
    background_tasks: BackgroundTasks,
    user: User | None = Depends(get_optional_user),
) -> DecideResult:
    persona = _compute_persona(request, user)
    skip_resolve = False
    try:
        if request.brand:
            result = await run_brand_price(request.query, request.brand)
        elif request.skip_intent_check:
            result = await run_single_debate(request.query, skip_clarify=True, persona=persona)
        else:
            # 정적 최종결과 캐시(decision_cache) 히트면 이미 캡처 시점에 한 번
            # resolve_lowest_price를 거친 링크라 - 다시 부르면 다나와에 실시간
            # 네트워크 호출(~0.5초)이 또 나가 캐시의 속도 이점이 사라진다.
            skip_resolve = decision_cache.lookup(request.query) is not None
            result = await run_debate(request.query, persona=persona)
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: 제안 전부 실패, ValueError: judge 응답에서 JSON을 못 찾음
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        # 외부 LLM API 오류 등 예상 못한 실패는 내부 정보를 노출하지 않고 502로 감싼다.
        raise HTTPException(
            status_code=502, detail="구매 결정을 처리하는 중 오류가 발생했습니다."
        ) from exc

    if not skip_resolve:
        result = await _resolve_danawa_urls(result)
    background_tasks.add_task(autocomplete.record_terms, _autocomplete_terms(request, result))
    return result


@app.post("/decide/stream")
async def decide_stream(
    request: DecideRequest, user: User | None = Depends(get_optional_user)
) -> StreamingResponse:
    """/decide와 같은 일을 하지만, 검색 완료·에이전트별 제안 완료·심사 단계마다
    한 줄씩(NDJSON) 흘려보낸다. 그래야 프론트가 세 에이전트를 다 기다리지 않고
    먼저 끝난 제안부터 화면에 보여줄 수 있다. 응답 헤더가 이미 200으로 나간
    뒤라 실패해도 HTTP 상태 코드를 바꿀 수 없으므로, 에러도 "error" 이벤트로
    흘려보낸다 — 프론트는 이 타입을 보고 에러 처리한다."""
    persona = _compute_persona(request, user)

    async def event_generator():
        try:
            if request.brand:
                result: DecideResult = await run_brand_price(request.query, request.brand)
                result = await _resolve_danawa_urls(result)
                yield json.dumps({"type": "final", "result": result.model_dump()}) + "\n"
            else:
                result = None
                # decide()와 같은 이유(skip_resolve 주석 참고) - decision_cache
                # 히트는 링크를 다시 해석하지 않는다.
                skip_resolve = not request.skip_intent_check and decision_cache.lookup(request.query) is not None
                stream = (
                    run_single_debate_stream(request.query, skip_clarify=True, persona=persona)
                    if request.skip_intent_check
                    else run_debate_stream(request.query, persona=persona)
                )
                async for event in stream:
                    if event["type"] == "final":
                        parsed = _decide_result_adapter.validate_python(event["result"])
                        result = parsed if skip_resolve else await _resolve_danawa_urls(parsed)
                        event["result"] = result.model_dump()
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
    """AI 상세검색(2026-08-12) - "음료수"처럼 짧고 애매한 검색어를 실제 검색
    상품명에 근거해 DeepSeek이 몇 가지 기준(브랜드/용량 등)으로 좁혀나가게
    제안한다.

    (2026-08-20) 첫 라운드 사전 체크로 프론트가 매 검색 전에 이 엔드포인트를
    먼저 부르던 용도는 없앴다 - /decide/stream이 내부적으로 타는 run_clarify()가
    이제 완전히 동일한 11번가 기반 facet 추출을 수행해 중복이었다(check_clarify_facets
    독스트링 참고). 지금은 SearchResults.tsx의 AI 상세검색 카드에서 사용자가
    자유 텍스트를 입력했을 때 facet을 실시간으로 재조회하는 용도로만 쓰인다 -
    base_query 드릴다운 재사용은 그 용도에서 애초에 안 쓰여서(항상 새 결합
    검색어로 부른다) 함께 제거했다."""
    persona = _compute_persona(request, user)
    return await check_clarify_facets(request.query, persona=persona)


@app.post("/decide/danawa-only", response_model=DecideResponse | BulkDecideResponse)
async def decide_danawa_only(request: DecideRequest) -> DecideResponse | BulkDecideResponse:
    """임시 실험 엔드포인트 - LLM 호출 0번(gpt/groq/deepseek 제안, judge 결정
    전부 생략), 다나와 실측 가격표만으로 규칙 기반 추천. LLM API 비용 절감
    목적의 로컬 테스트 경로라 /decide와 별도로 둔다 - 프론트엔드는 아직
    이 경로를 쓰지 않는다. 검색어가 서로 다른 상품에 걸쳐 있으면(예: "노트북")
    DecideResponse 대신 BulkDecideResponse(후보 목록)를 반환한다."""
    try:
        return await run_danawa_only_debate(request.query, base_query=request.base_query)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="다나와 전용 처리 중 오류가 발생했습니다."
        ) from exc


@app.post("/decide/danawa-only/stream")
async def decide_danawa_only_stream(request: DecideRequest) -> StreamingResponse:
    """/decide/danawa-only의 스트리밍 버전 - 사용자 요청(2026-08-11, "1개 서치
    완료되면 1개 올려줘 먼저"). SSE(text/event-stream)로 후보가 끝나는 대로
    {"type": "candidate", ...}를 내보내고, 마지막에 {"type": "final", ...}
    (또는 실패 시 {"type": "error", ...})를 내보낸다.

    이미 200으로 스트림을 연 뒤라 HTTP 상태코드로 실패를 알릴 수 없다 - 그래서
    /decide/danawa-only와 달리 502를 던지지 않고 "error" 이벤트로 실패를
    알린다(프론트가 이벤트 타입으로 구분해서 처리)."""

    async def _events():
        try:
            async for event in run_danawa_only_debate_stream(request.query, base_query=request.base_query):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            error_event = {"type": "error", "message": "다나와 전용 처리 중 오류가 발생했습니다."}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")
