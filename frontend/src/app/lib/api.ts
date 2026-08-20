export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export type AgentName = 'gpt' | 'groq' | 'deepseek';

export interface Proposal {
  agent: AgentName;
  product_name: string | null;
  price: string | null;
  retailer: string | null;
  url: string | null;
  reasoning: string | null;
  error: string | null;
  verified?: boolean | null;
  challenge_note?: string | null;
  proposed_by?: AgentName[] | null;
}

export interface Decision {
  product_name: string;
  price: string;
  retailer: string;
  url: string;
  reasoning: string;
  chosen_agent: AgentName;
}

export interface StyleGuideGroup {
  label: string;
  description: string;
  // proposals 중 하나의 url과 정확히 일치한다(그라운딩) - 백엔드
  // adk_pipeline._build_style_guide가 이미 검증해서 보낸다.
  url: string;
}

export interface StyleGuide {
  intro: string;
  groups: StyleGuideGroup[];
  closing_pick: string | null;
}

export interface DecideResponse {
  mode: 'single';
  query: string;
  proposals: Proposal[];
  decision: Decision;
  // 취향 주도 카테고리(패션의류/잡화 등)에서만 채워진다 - 없으면 기존
  // 단일 추천 UI 그대로.
  style_guide?: StyleGuide | null;
}

export interface BrandOption {
  brand: string;
  product_name: string;
  price: string;
  retailer: string;
  url: string;
  reasoning?: string | null;
  delivery_note?: string | null;
}

export interface BulkProposal {
  agent: AgentName;
  options: BrandOption[];
  error: string | null;
}

export interface BulkDecision {
  options: BrandOption[];
  reasoning: string;
}

export interface PriceRange {
  min: string;
  max: string;
}

export interface BulkDecideResponse {
  mode: 'bulk';
  query: string;
  proposals: BulkProposal[];
  decision: BulkDecision;
  price_range: PriceRange | null;
}

export interface ClarifyFacet {
  label: string;
  options: string[];
  // 다른 facet(어느 것이든 - 브랜드로 한정 안 됨)에서 뭘 고르면, 추가 요청
  // 없이 이 매핑으로 이 facet의 보이는 옵션을 즉시 좁힌다(2026-08-14 일반화:
  // 브랜드="삼성전자" -> 시리즈가 갤럭시 계열만이었던 걸, 시리즈="초코파이
  // 바나나" -> 용량이 그 시리즈에 실제로 있는 값만으로도 확장). 키는 다른
  // facet의 옵션 문자열, 값은 그 선택이 주어졌을 때 이 facet에서 유효한 옵션들.
  options_by_selection?: Record<string, string[]> | null;
}

export interface ClarifyOptions {
  brands: string[];
  products: string[];
  volumes: string[];
  quantities: string[];
  facets: ClarifyFacet[];
  // 사용자에게 고르라고 묻지 않고 Groq이 질의만 보고 자동으로 매핑한 카테고리
  // (app.category.classify_category) - 클릭 옵션이 아니라 정보성 값이다.
  detected_category: string | null;
}

export interface ClarifyResponse {
  mode: 'clarify';
  query: string;
  options: ClarifyOptions;
}

export interface BrandPriceResponse {
  mode: 'brand_price';
  query: string;
  brand: string;
  option: BrandOption | null;
  error: string | null;
}

export type DecideResult =
  | DecideResponse
  | BulkDecideResponse
  | ClarifyResponse
  | BrandPriceResponse;

export interface OcrResult {
  text: string;
  confidence: number | null;
  latency_ms: number | null;
  block_count: number;
  error: string | null;
}

export interface OcrCleanupResult {
  cleaned_text: string | null;
  search_query: string | null;
  notes: string | null;
  error: string | null;
}

export interface OcrExtractResponse {
  ocr: OcrResult;
  cleaned: OcrCleanupResult | null;
}

export class ApiError extends Error {}

// AI 상세검색(2026-08-12) - "음료수"처럼 짧고 애매한 검색어를 11번가 검색 결과에
// 근거해 DeepSeek이 카테고리/브랜드/용량 같은 기준(facet)으로 좁혀나가도록 제안한다.
// 백엔드가 needs_clarification()으로 한 번 더 걸러서, 명확한 검색어면 검색/LLM
// 호출 없이 즉시 facets: []로 끝난다(app/debate.py::check_clarify_facets 참고).
//
// looksAmbiguous()는 그 백엔드 검사(backend/app/intent.py::_is_short_bare_query)를
// 그대로 흉내낸 순수 클라이언트 프리필터다 - 명백히 구체적인 검색어(숫자가 있거나
// 단어가 5개 이상)에 대해서는 이 API를 아예 호출하지 않아서, 거의 모든 검색에서
// 왕복 하나조차 안 생기게 한다. 오탐(애매한데 여기서 걸러짐)이 있어도 위험하지
// 않다 - 그런 경우 사용자는 그냥 원래 검색 경로로 바로 넘어갈 뿐이다.
//
// 2 -> 4(2026-08-15, 카테고리별 메타데이터 차이 - "냉장고 살 때랑 콜라 살 때
// 쓰는 메타데이터가 다르다") - intent.py::SHORT_QUERY_TOKEN_LIMIT과 함께 넓혀서
// "삼성 냉장고 스탠드형"처럼 짧지만 구체성이 붙기 시작한 질의도 AI 상세검색
// (카테고리별 동적 facet)을 먼저 시도하도록 한다.
const HAS_DIGIT_PATTERN = /\d/;

export function looksAmbiguous(query: string): boolean {
  const trimmed = query.trim();
  if (!trimmed) return false;
  const tokens = trimmed.split(/\s+/);
  return tokens.length <= 4 && !HAS_DIGIT_PATTERN.test(trimmed);
}

// baseQuery(2026-08-13, "조금 더 빠르게" 요청) - 드릴다운 중(예: "핸드폰" ->
// "핸드폰 삼성전자")이면 그 체인의 맨 처음 검색어를 실어 보낸다. 백엔드가 그
// base_query 결과를 검색해 나머지는 로컬 필터링만 하게 해준다
// (app/debate.py::check_clarify_facets).
//
// sessionPreferences/token(2026-08-15, 사용자 페르소나 기반 상품 매핑) - 이번
// 세션에서 지금까지 고른 facet 값({라벨: 값})을 실어 보내면, 로그인 계정의
// 영구 선호도(app.preferences)와 백엔드가 병합해 facet 옵션 순서에 소프트하게
// 반영한다(옵션을 줄이거나 지어내지 않고, 이미 있는 값을 맨 앞으로 당기기만
// 한다). token이 없으면 세션 값만 반영되고 계정 조회는 건너뛴다.
export async function checkClarifyFacets(
  query: string,
  baseQuery?: string,
  sessionPreferences?: Record<string, string>,
  token?: string | null
): Promise<ClarifyResponse> {
  const response = await fetch(`${API_URL}/decide/clarify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      query,
      ...(baseQuery ? { base_query: baseQuery } : {}),
      ...(sessionPreferences && Object.keys(sessionPreferences).length > 0
        ? { session_preferences: sessionPreferences }
        : {}),
    }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail || `요청이 실패했습니다 (${response.status})`);
  }

  return response.json();
}

// 사용자 페르소나 기록(2026-08-15) - 로그인한 사용자가 clarify에서 facet/브랜드
// 값을 하나 고를 때마다 fire-and-forget으로 호출해 계정에 누적한다. 실패해도
// 검색 흐름을 막지 않는다(다음 checkClarifyFacets 호출이 그냥 이번 선택을
// 반영 못 할 뿐).
export async function recordPreference(token: string, label: string, value: string): Promise<void> {
  try {
    await fetch(`${API_URL}/preferences`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ label, value }),
    });
  } catch {
    // 조용히 무시 - 페르소나는 있으면 좋은 부가 기능이지 필수 경로가 아니다.
  }
}

export async function decide(query: string, brand?: string): Promise<DecideResult> {
  const response = await fetch(`${API_URL}/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(brand ? { query, brand } : { query }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail || `요청이 실패했습니다 (${response.status})`);
  }

  return response.json();
}

export type DecideStage = 'searching';

export type DecideStreamEvent =
  | { type: 'status'; stage: DecideStage }
  | { type: 'proposal'; proposal: Proposal }
  | { type: 'final'; result: DecideResult }
  | { type: 'error'; message: string };

/** /decide와 같은 일을 하지만, 서버가 단계별로 흘려보내는 NDJSON(줄바꿈으로 구분된 JSON)을
 * 그때그때 onEvent로 넘겨준다 — 세 에이전트를 다 기다리지 않고 먼저 끝난 제안부터 보여줄 수 있다.
 *
 * baseQuery(2026-08-20, HITL 구조적 필터 재설계) - 드릴다운 후속 턴(예: "핸드폰"
 * -> "핸드폰 삼성전자")이면 그 체인의 맨 처음 검색어를 실어 보낸다. 안 보내면
 * 백엔드가 매번 재구성된 전체 문자열로 새로 검색한다 - 보내면 base_query로
 * 검색한 뒤 나머지는 로컬 필터링만 해서(app.debate._search_candidates) 같은
 * 검색을 중복으로 다시 하지 않는다. */
export async function decideStream(
  query: string,
  onEvent: (event: DecideStreamEvent) => void,
  brand?: string,
  signal?: AbortSignal,
  skipIntentCheck?: boolean,
  baseQuery?: string
): Promise<void> {
  const response = await fetch(`${API_URL}/decide/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      ...(brand ? { brand } : {}),
      ...(skipIntentCheck ? { skip_intent_check: true } : {}),
      ...(baseQuery ? { base_query: baseQuery } : {}),
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail || `요청이 실패했습니다 (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex = buffer.indexOf('\n');
    while (newlineIndex !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) onEvent(JSON.parse(line) as DecideStreamEvent);
      newlineIndex = buffer.indexOf('\n');
    }
  }
}

export async function extractOcr(file: File): Promise<OcrExtractResponse> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_URL}/ocr/extract`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail || `이미지 분석에 실패했습니다 (${response.status})`);
  }

  return response.json();
}

const FALLBACK_CLARIFY_QUESTION = '몇 가지 후보를 찾았어요 — 아래에서 골라주시겠어요?';

/** 이번 라운드에 물어볼 축(브랜드/제품/용량/개수)의 후보들을 실제 상담원처럼
 * 자연스러운 질문 문장으로 바꿔달라고 서버(GPT)에 요청한다 — "브랜드를
 * 선택하면 좁혀드려요" 같은 고정 라벨 대신 채팅 말풍선으로 먼저 보여줄 문장. */
export async function askClarifyQuestion(query: string, options: string[]): Promise<string> {
  try {
    const response = await fetch(`${API_URL}/clarify/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, options }),
    });
    if (!response.ok) return FALLBACK_CLARIFY_QUESTION;
    const data: { message: string } = await response.json();
    return data.message || FALLBACK_CLARIFY_QUESTION;
  } catch {
    return FALLBACK_CLARIFY_QUESTION;
  }
}

export async function fetchAutocomplete(query: string, signal?: AbortSignal): Promise<string[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];

  const response = await fetch(`${API_URL}/autocomplete?q=${encodeURIComponent(trimmed)}`, { signal });
  if (!response.ok) return [];
  return response.json();
}

// --- 로그인 계정별 검색 기록 (서버에 저장, 기기 상관없이 동일 계정이면 같은 기록) ---

export interface ServerHistoryEntry {
  id: string;
  query: string;
  timestamp: number; // 서버는 epoch seconds(float)로 내려준다
  result: DecideResult;
}

export async function fetchServerHistory(token: string): Promise<ServerHistoryEntry[]> {
  const response = await fetch(`${API_URL}/history`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) return [];
  return response.json();
}

export async function saveServerHistory(
  token: string,
  query: string,
  result: DecideResult
): Promise<ServerHistoryEntry | null> {
  const response = await fetch(`${API_URL}/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ query, result }),
  });
  if (!response.ok) return null;
  return response.json();
}

export async function deleteServerHistoryEntry(token: string, id: string): Promise<void> {
  await fetch(`${API_URL}/history/${id}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function clearServerHistory(token: string): Promise<void> {
  await fetch(`${API_URL}/history`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  });
}
