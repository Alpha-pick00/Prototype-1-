from typing import Literal

from pydantic import BaseModel

# "danawa": PART 4-2(2026-08-11 지시서) - 다나와 A등급 최저가 후보가 judge의
# 선택 대상 풀에 직접 들어간다. 프론트엔드는 AGENT_LABEL[agent] || agent로
# 렌더링해(frontend/src/app/components/SearchResults.tsx) 모르는 값이 와도
# 원문 그대로 표시할 뿐 깨지지 않는다 - 확인 후 추가했다.
AgentName = Literal["gpt", "groq", "deepseek", "danawa", "elevenst"]
AuthProvider = Literal["google", "kakao", "naver"]


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float | None = None


class Proposal(BaseModel):
    agent: AgentName
    product_name: str | None = None
    price: str | None = None
    retailer: str | None = None
    url: str | None = None
    reasoning: str | None = None
    error: str | None = None
    verified: bool | None = None
    challenge_note: str | None = None
    proposed_by: list[AgentName] | None = None


class AgentCandidate(BaseModel):
    product_name: str
    price_krw: int | None = None
    retailer: str | None = None
    url: str | None = None
    reasoning: str | None = None


class RefinedQuery(BaseModel):
    query: str
    error: str | None = None


class ChallengeVerdict(BaseModel):
    url: str | None = None
    verified: bool
    note: str = ""
    # 2026-08-19("실시간으로 판매처 페이지를... 가져오지 않는 이상") - challenge
    # 직전 _ExtractPagesNode가 이미 후보 페이지를 라이브로 재조회해두는데, 그동안
    # 이 재조회 원문이 그라운딩 검증(verified/note)에만 쓰이고 가격 자체는
    # 갱신되지 않았다 - 이미 지불한 네트워크 비용을 가격 신선도에도 활용한다.
    # 재조회 원문에서 명확히 확인 안 되면(추측 금지) None으로 둔다.
    refreshed_price_krw: int | None = None


class ChallengeResult(BaseModel):
    verdicts: list[ChallengeVerdict] = []
    error: str | None = None


class Decision(BaseModel):
    product_name: str
    price: str
    retailer: str
    url: str
    reasoning: str
    chosen_agent: AgentName
    # "danawa_offer": app.price_table이 다나와 실측 가격표의 A등급(링크 생성
    # 가능) offer와 대조해 price/url을 검증된 값으로 교체했다는 뜻.
    # "elevenst_offer": 11번가 오픈 API(ProductSearch) 응답을 그대로 썼다는 뜻 -
    # 다나와처럼 이미 1st-party 구조화 데이터라 LLM 추정이 섞이지 않는다.
    # "llm_guess"(기본값): 그런 대조 없이 LLM이 제안한 값 그대로.
    price_source: Literal["danawa_offer", "elevenst_offer", "llm_guess"] = "llm_guess"
    # 최종 선택된 후보가 DeepSeek challenge 검증을 통과했는지(Proposal.verified와
    # 같은 의미) — judge 경로는 매칭된 proposal의 값을 그대로 물려받고, relaxed
    # fallback 경로(2026-08-16 강화)는 별도로 challenge를 태워 채운다. None은
    # "검증 자체가 안 됐거나 실패함"(다나와 실측처럼 애초에 challenge가 필요
    # 없는 경우도 apply_challenge에서 True로 강제되므로 여기 None에는 안 걸림).
    verified: bool | None = None


class JudgeVerdict(BaseModel):
    """judge LlmAgent의 output_schema — chosen_agent 없이 선택한 상품만 반환하고,
    실제 chosen_agent는 adk_pipeline이 url로 역매칭해서 채운다(제안자가 여럿일
    수 있어 LLM에게 단일 리터럴을 직접 고르게 하지 않는다)."""

    product_name: str
    price: str
    retailer: str
    url: str
    reasoning: str


class DecideRequest(BaseModel):
    query: str
    brand: str | None = None
    # Human-in-the-loop으로 브랜드/용량/개수를 이미 하나 골라 검색어에 이어붙여
    # 재검색하는 요청이면 True — is_bulk_query()/needs_clarification() 같은
    # "첫 질의가 애매한지" 판단용 휴리스틱을 건너뛴다. 이미 특정 상품을 좁혀가는
    # 중인데, 예컨대 "80ml"처럼 용량이 붙은 재검색어가 새 대량구매 질의로
    # 오판되는 걸 막기 위함.
    skip_intent_check: bool = False
    # /decide/clarify 전용(app.debate.check_clarify_facets) - AI 상세검색을
    # 여러 턴에 걸쳐 좁혀나갈 때(예: "핸드폰" -> "핸드폰 삼성전자") 매 라운드마다
    # search.danawa.com을 새로 때리면 10초 Crawl-delay가 매번 붙어 느리다.
    # base_query에 그 드릴다운의 맨 처음 검색어(이미 캐시됐을 가능성이 높다)를
    # 넘기면, 백엔드가 그걸로 검색해 캐시를 재사용하고 나머지는 로컬 필터링만
    # 한다 - 다른 엔드포인트는 이 필드를 무시한다.
    base_query: str | None = None
    # /decide/clarify 전용(2026-08-15, 사용자 페르소나 기반 상품 매핑) - 로그인
    # 여부와 무관하게 "이번 세션에서 지금까지 고른 값들"을 {facet 라벨: 값}으로
    # 프론트가 누적해 매 요청에 실어 보낸다. 로그인 계정의 영구 선호도
    # (app.preferences)와 병합해 facet 옵션 순서에 소프트하게 반영한다 - 세션
    # 값이 계정 값보다 최신이라 우선한다.
    session_preferences: dict[str, str] | None = None


class PreferenceRecordRequest(BaseModel):
    label: str
    value: str


class PriceTableOffer(BaseModel):
    seller: str
    price_krw: int
    delivery_text: str | None = None
    domain: str | None = None
    trust: float | None = None
    linkable: bool
    rank: int


class PriceTable(BaseModel):
    source: str = "danawa"
    source_pcode: str | None = None
    product_name: str | None = None
    offers: list[PriceTableOffer]
    spread: float | None = None
    # B-3a 실측: 다나와 상세페이지는 판매처 최대 10개만 정적 HTML에 노출한다.
    # 검색결과 페이지의 "N몰" 표기(있을 때만)로 total_mall_count가 채워지면,
    # 그게 offers_shown보다 클 때 is_partial=True - "최저가"가 아니라
    # "확인된 최저가"라는 뜻이고 price_label에 그대로 반영된다.
    total_mall_count: int | None = None
    offers_shown: int = 0
    is_partial: bool = False
    price_label: str = "최저가"


class StyleGuideGroup(BaseModel):
    label: str
    description: str
    # 반드시 proposals 중 하나의 url과 정확히 일치해야 한다(그라운딩) -
    # adk_pipeline._build_style_guide가 실제 후보에 없는 url은 걸러낸다.
    url: str


class StyleGuide(BaseModel):
    intro: str
    groups: list[StyleGuideGroup]
    closing_pick: str | None = None


class DecideResponse(BaseModel):
    mode: Literal["single"] = "single"
    query: str
    proposals: list[Proposal]
    decision: Decision
    price_table: PriceTable | None = None
    # 취향 주도 카테고리(패션의류/잡화 등)에서만 채워진다(2026-08-19 사용자
    # 요청: GPT 쇼핑처럼 스타일별로 여러 검증된 후보를 묶어 보여주되, 최종
    # 추천(decision)은 지금처럼 하나로 유지). None이면 기존 단일 추천
    # 응답과 완전히 동일 - 하위 호환.
    style_guide: StyleGuide | None = None


class BrandOption(BaseModel):
    brand: str
    product_name: str
    price: str
    retailer: str
    url: str
    reasoning: str | None = None
    delivery_note: str | None = None


class BulkProposal(BaseModel):
    agent: AgentName
    options: list[BrandOption] = []
    error: str | None = None


class BulkDecision(BaseModel):
    options: list[BrandOption]
    reasoning: str


class PriceRange(BaseModel):
    min: str
    max: str


class BulkDecideResponse(BaseModel):
    mode: Literal["bulk"] = "bulk"
    query: str
    proposals: list[BulkProposal]
    decision: BulkDecision
    price_range: PriceRange | None = None


class ClarifyFacet(BaseModel):
    """brands/volumes/quantities는 원래 GPT가 고정된 3종류로만 뽑아내던 값이고,
    facets는 그 외에 DeepSeek이 검색어별로 자유롭게 뽑아내는 임의의 기준(예: 카테고리,
    제조사, 용기형태, 특징)이다 - 라벨 자체도 고정돼 있지 않아 list[dict] 형태로 둔다."""

    label: str
    options: list[str] = []
    # 다른 어떤 facet의 값이든 - 브랜드 한정이 아니다(app.debate._attach_facet_crossfilter,
    # 2026-08-14: "시리즈에 초코파이 바나나를 골랐다면 용량에 없는것들은 선택할수
    # 없게"). 키는 "다른 facet에서 고를 수 있는 옵션 문자열"이고, 값은 그 선택이
    # 주어졌을 때 이 facet에서 실제로 유효한(같은 상품명에 같이 등장하는) 옵션들이다.
    # 예: 시리즈="초코파이 바나나" -> 용량이 [468g, ...]만. 프론트는 지금까지 고른
    # 값 전부를 이 매핑에 돌려 교집합으로 보이는 옵션을 좁힌다. 해당 키가 없으면
    # options 전체를 보여준다.
    options_by_selection: dict[str, list[str]] | None = None


class ClarifyOptions(BaseModel):
    brands: list[str] = []
    products: list[str] = []
    volumes: list[str] = []
    quantities: list[str] = []
    facets: list[ClarifyFacet] = []


class ClarifyResponse(BaseModel):
    mode: Literal["clarify"] = "clarify"
    query: str
    options: ClarifyOptions


class ClarifyAskRequest(BaseModel):
    query: str
    options: list[str]


class ClarifyAskResponse(BaseModel):
    message: str


class BrandPriceResponse(BaseModel):
    mode: Literal["brand_price"] = "brand_price"
    query: str
    brand: str
    option: BrandOption | None = None
    error: str | None = None


DecideResultUnion = DecideResponse | BulkDecideResponse | ClarifyResponse | BrandPriceResponse


class OcrResult(BaseModel):
    text: str = ""
    confidence: float | None = None
    latency_ms: int | None = None
    block_count: int = 0
    error: str | None = None


class OcrCleanupResult(BaseModel):
    cleaned_text: str | None = None
    search_query: str | None = None
    notes: str | None = None
    error: str | None = None


class OcrExtractResponse(BaseModel):
    ocr: OcrResult
    cleaned: OcrCleanupResult | None = None


class User(BaseModel):
    provider: AuthProvider
    provider_user_id: str
    email: str | None = None
    name: str | None = None
    picture: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: User


class GoogleAuthRequest(BaseModel):
    access_token: str  # Google OAuth2 토큰 클라이언트(팝업)가 내려주는 access token


class OAuthCodeRequest(BaseModel):
    code: str
    redirect_uri: str
    state: str | None = None  # 네이버는 토큰 교환 시 state가 필요하다


class HistoryEntry(BaseModel):
    id: str
    query: str
    timestamp: float
    result: DecideResultUnion


class SaveHistoryRequest(BaseModel):
    query: str
    result: DecideResultUnion
