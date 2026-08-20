import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  checkClarifyFacets,
  decide,
  decideStream,
  extractOcr,
  fetchServerHistory,
  saveServerHistory,
  deleteServerHistoryEntry,
  clearServerHistory,
  looksAmbiguous,
  recordPreference,
  ApiError,
  type DecideResult,
  type DecideStage,
  type Proposal,
  type ServerHistoryEntry,
} from '../lib/api';
import { getStoredToken } from '../lib/auth';
import { useAuth } from './AuthContext';
import {
  deleteHistoryEntry,
  clearHistory,
  loadHistory,
  saveHistoryEntry,
  type HistoryEntry,
} from '../lib/history';
import type { ClarifyStep } from '../components/SearchResults';

const fromServerEntry = (entry: ServerHistoryEntry): HistoryEntry => ({
  id: entry.id,
  query: entry.query,
  timestamp: entry.timestamp * 1000, // 서버는 초 단위, 프론트는 Date.now() 기준 ms 단위로 통일
  result: entry.result,
});

export type TurnStatus = 'loading' | 'result' | 'error';

// ChatGPT/Claude 스타일 대화 스레드의 한 왕복(사용자 메시지 -> AI 오케스트레이션 답변).
// displayQuery는 사용자 말풍선에 그대로 보여줄 텍스트, requestQuery/brand는
// 실제로 서버에 던진 값 - 브랜드 선택 턴은 이 둘이 다르다(말풍선엔 "삼성"만
// 보이지만 실제 요청은 원래 검색어+브랜드 조합이라 재시도 시 필요).
export interface ChatTurn {
  id: string;
  displayQuery: string;
  requestQuery: string;
  brand?: string;
  // AI 상세검색 드릴다운 체인의 맨 처음 검색어(속도 개선, 2026-08-13) - "핸드폰"
  // -> "핸드폰 삼성전자"로 좁혀가는 동안 이 값은 계속 "핸드폰"으로 고정된다.
  // checkClarifyFacets가 이걸 base_query로 보내 백엔드 캐시를 재사용한다.
  baseQuery: string;
  status: TurnStatus;
  result: DecideResult | null;
  errorMessage: string;
  // AI 오케스트레이션(adk_pipeline: 정제→검색→제안→검증→심사) 진행 상태 -
  // decideStream이 이 턴을 처리하는 동안 status/proposal 이벤트로 채워진다.
  streamingStage: DecideStage | null;
  streamingProposals: Proposal[];
  // 메시지 시간 표시(사용자 요청, "클로드 너처럼 날짜기능") - epoch ms.
  // loadFromHistory는 실제 기록 시각을 쓰고, 그 외엔 턴 생성 시각.
  createdAt: number;
}

// ChatGPT/Gemini처럼 "창(대화)" 하나에 여러 턴이 계속 이어지고, 새 상품을 검색할
// 때는 새 창을 연다(2026-08-15, "새로운 상품을 검색할 때는 새로운 창을 띄워서
// 검색할 수 있게하고 하나의 창에서는 하나의 대화로그가 계속 이어질수 있도록").
// 세션 범위로만 유지한다(새로고침하면 사라짐) - 완료된 검색 결과 자체는
// 기존 history(HistoryEntry, 서버/로컬 영구 저장)가 이미 별도로 보존한다.
export interface Conversation {
  id: string;
  // 사이드바 목록에 보여줄 제목 - 이 대화의 첫 턴 displayQuery로 고정한다
  // (이후 턴이 쌓여도 안 바뀜, ChatGPT의 대화 제목과 같은 동작).
  title: string;
  turns: ChatTurn[];
  updatedAt: number;
}

interface SearchContextValue {
  turns: ChatTurn[];
  isBusy: boolean;
  ocrBusy: boolean;
  history: HistoryEntry[];
  // 사용자 페르소나(2026-08-15) - 이번 세션에서 지금까지 고른 {facet 라벨: 값}.
  // SearchResults가 옵션 버튼에 "선호" 표시를 붙이는 데 쓴다 - 실제 옵션 순서
  // 반영(우선순위를 앞으로 당기는 것)은 백엔드(check_clarify_facets)가 이미
  // 하므로, 여기서는 순수하게 시각적 표시 용도다.
  sessionPreferences: Record<string, string>;
  // 최종 결과 카드의 "다른 관점에서 보기" 칩(2026-08-19, "가성비/성분 선호가
  // 개인마다 다르니 페르소나로 반영하면 좋겠다") - facet 선택과 같은 방식으로
  // 세션/계정 페르소나에 기록한다. 값 자체는 실제 클릭한 proposal에서 그대로
  // 가져와야 한다(지어낸 라벨을 기록하면 다음 검색에 근거 없는 선호가 반영됨).
  rememberPreference: (label: string, value: string) => void;
  sendMessage: (q: string) => Promise<void>;
  selectBrand: (turnId: string, brand: string) => Promise<void>;
  selectFacets: (turnId: string, selected: Record<string, string>) => Promise<void>;
  selectClarifyOption: (turnId: string, step: Exclude<ClarifyStep, 'brand'>, value: string) => Promise<void>;
  retryTurn: (turnId: string) => Promise<void>;
  editTurn: (turnId: string, newQuery: string) => Promise<void>;
  handleImageUpload: (file: File) => Promise<void>;
  handleReset: () => void;
  loadFromHistory: (entry: HistoryEntry) => void;
  deleteFromHistory: (id: string) => void;
  clearAllHistory: () => void;
}

const SearchContext = createContext<SearchContextValue | null>(null);

// facet/옵션 선택을 이어붙일 때 이미 있는 단어를 또 붙이지 않는다(사용자 요청,
// 2026-08-14: "다나와에서 '초코파이 오리온 초코파이 바나나 468g'에 대한 가격
// 정보를 찾지 못했다" - 시리즈 옵션 "초코파이 바나나" 자체가 이미 원래 검색어
// "초코파이"를 포함하고 있어서, 그냥 이어붙이면 "초코파이"가 두 번 들어가
// 검색이 이상하게 안 맞는 검색어가 됐다). 토큰(공백 기준) 단위로만 비교한다.
export const dedupeAppend = (base: string, addition: string): string => {
  const baseTokens = base.trim().split(/\s+/).filter(Boolean);
  const seen = new Set(baseTokens.map((t) => t.toLowerCase()));
  const newTokens = addition
    .trim()
    .split(/\s+/)
    .filter((t) => t && !seen.has(t.toLowerCase()));
  return [...baseTokens, ...newTokens].join(' ');
};

// 고정 축(product/volume/quantity) -> 사용자 페르소나 라벨 매핑. AI
// 상세검색(facet)과 서로 다른 라벨 체계를 쓰므로("volume" vs 실제 facet 라벨
// "용량"), 향후 facet 재정렬에서 실제로 매칭될 수 있는 축(용량)만 공용 라벨로
// 옮겨 기록한다. product는 특정 상품명 그 자체라 재사용 가치가 낮아 기록하지 않는다.
const CLARIFY_STEP_PERSONA_LABEL: Partial<Record<Exclude<ClarifyStep, 'brand'>, string>> = {
  volume: '용량',
  quantity: '구매유형',
};

const newTurn = (
  displayQuery: string,
  requestQuery: string,
  brand?: string,
  baseQuery?: string
): ChatTurn => ({
  id: crypto.randomUUID(),
  displayQuery,
  requestQuery,
  brand,
  baseQuery: baseQuery || requestQuery,
  status: 'loading',
  result: null,
  errorMessage: '',
  streamingStage: null,
  streamingProposals: [],
  createdAt: Date.now(),
});

export const SearchProvider = ({ children }: { children: React.ReactNode }) => {
  const { user } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [ocrBusy, setOcrBusy] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  // 사용자 페르소나(2026-08-15, "냉장고 살 때랑 콜라 살 때 쓰는 메타데이터가
  // 다르다" -> "사용자 페르소나 기반으로 상품 매핑") - 이번 세션에서 지금까지
  // 고른 facet/축 값을 {라벨: 값}으로 누적한다. 대화(conversation) 하나에
  // 갇히지 않고 세션 전체(예: 폰 대화에서 "삼성" 고른 뒤, 완전히 새로 연
  // 이어폰 대화에도)에 걸쳐 유지된다. 로그인 계정에는 추가로 영구 저장한다
  // (rememberPreference 참고) - 세션 값은 새로고침하면 사라지지만, 계정 값은
  // 다음 방문에도 /decide/clarify가 자동으로 다시 불러와 반영한다.
  const [sessionPreferences, setSessionPreferences] = useState<Record<string, string>>({});

  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;
  const turns = activeConversation?.turns ?? [];

  // 로그인 상태가 바뀌면 기록 소스를 전환한다 — 로그인하면 그 계정의 서버 기록을
  // 불러오고, 로그아웃하면 이 브라우저의 로컬 기록으로 되돌아간다.
  useEffect(() => {
    if (!user) {
      setHistory(loadHistory());
      return;
    }
    const token = getStoredToken();
    if (!token) return;
    fetchServerHistory(token).then((entries) => setHistory(entries.map(fromServerEntry)));
  }, [user]);

  const persistHistoryEntry = async (q: string, data: DecideResult) => {
    if (user) {
      const token = getStoredToken();
      if (!token) return;
      const saved = await saveServerHistory(token, q, data);
      if (saved) {
        setHistory((prev) => [fromServerEntry(saved), ...prev]);
      }
      return;
    }
    setHistory(saveHistoryEntry(q, data));
  };

  // 턴 ID는 crypto.randomUUID()로 전역 유일하므로, 어느 대화가 활성 상태인지와
  // 무관하게 모든 대화를 뒤져 실제로 그 턴을 담고 있는 대화만 갱신한다 - 스트리밍
  // 응답이 도착하는 동안 사용자가 다른 대화로 전환해도 엉뚱한(현재 활성) 대화가
  // 아니라 turn이 실제로 속한 대화가 정확히 갱신된다.
  const patchTurn = (id: string, patch: Partial<ChatTurn>) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.turns.some((t) => t.id === id)
          ? { ...c, turns: c.turns.map((t) => (t.id === id ? { ...t, ...patch } : t)), updatedAt: Date.now() }
          : c
      )
    );
  };

  const appendStreamingProposal = (turnId: string, proposal: Proposal) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.turns.some((t) => t.id === turnId)
          ? {
              ...c,
              turns: c.turns.map((t) =>
                t.id === turnId ? { ...t, streamingProposals: [...t.streamingProposals, proposal] } : t
              ),
            }
          : c
      )
    );
  };

  const findTurn = (turnId: string): ChatTurn | undefined =>
    conversations.flatMap((c) => c.turns).find((t) => t.id === turnId);

  // 새 턴을 대화에 붙인다 - conversationId가 없으면(새 대화의 첫 턴) 새 대화를
  // 만들어 목록 맨 앞에 얹고 활성으로 전환한다. 이게 "새로운 상품을 검색할 때는
  // 새로운 창"의 실제 진입점: handleReset이 activeConversationId를 null로
  // 돌려놓으면, 다음 sendMessage가 이 분기를 타 자동으로 새 창을 연다.
  const appendTurn = (conversationId: string | null, turn: ChatTurn) => {
    if (conversationId) {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === conversationId ? { ...c, turns: [...c.turns, turn], updatedAt: Date.now() } : c
        )
      );
      return conversationId;
    }
    const id = crypto.randomUUID();
    setConversations((prev) => [{ id, title: turn.displayQuery, turns: [turn], updatedAt: Date.now() }, ...prev]);
    setActiveConversationId(id);
    return id;
  };

  // 페르소나 한 줄(라벨:값) 기록 - 세션 상태는 즉시 반영해 바로 다음
  // checkClarifyFacets 호출부터 쓰이고, 로그인 계정에는 fire-and-forget으로
  // 영구 저장한다(실패해도 검색 흐름에 영향 없음).
  const rememberPreference = (label: string, value: string) => {
    setSessionPreferences((prev) => ({ ...prev, [label]: value }));
    if (user) {
      const token = getStoredToken();
      if (token) recordPreference(token, label, value).catch(() => {});
    }
  };

  // 턴 하나의 실제 검색/조회를 실행하고 그 턴만 갱신한다. sendMessage(새 턴 추가)/
  // selectBrand/selectFacets/selectClarifyOption(후속 턴 추가)/retryTurn(기존 턴
  // 재실행) 전부 이 위에서 돈다. AI 오케스트레이션(adk_pipeline, decideStream)이
  // 이 함수 안쪽(백엔드 API 호출)에서 한 턴을 처리하는 stateless 단계를 맡는다 -
  // 턴/대화/baseQuery 관리는 이 함수 바깥(sendMessage 등 호출부)의 책임이다.
  //
  // skipIntentCheck - 이미 브랜드를 골랐거나(brand) 이전 턴에서 축적된 검색어로
  // 이어가는 드릴다운 후속 턴(requestQuery !== baseQuery)이면 백엔드의 내부
  // clarify 재판정(needs_clarification)을 건너뛴다 - 안 그러면 이미 한 번 답한
  // 축(브랜드/용량 등)에 대해 서버가 또 clarify를 띄우는 재질문 버그가 생긴다
  // (2026-08 통합 병합 승인안의 "skip_internal_clarify" 요구사항을 기존
  // skip_intent_check 플래그로 구현).
  //
  // personaOverride - 바로 이 턴을 만든 선택(예: 방금 rememberPreference로 기록한
  // 라벨:값)을 checkClarifyFacets 호출에 즉시 반영하기 위한 값이다. setState는
  // 비동기라 rememberPreference 직후 곧바로 runTurn을 불러도 이 함수가 캡처한
  // sessionPreferences는 아직 리렌더 전 값(반영 안 됨)일 수 있어, 호출부가 방금
  // 배운 값을 명시적으로 얹어 보낸다.
  const runTurn = async (
    id: string,
    requestQuery: string,
    brand?: string,
    baseQuery?: string,
    personaOverride?: Record<string, string>
  ) => {
    if (brand) {
      try {
        const data = await decide(requestQuery, brand);
        patchTurn(id, { status: 'result', result: data });
        persistHistoryEntry(requestQuery, data).catch(() => {});
      } catch (err) {
        patchTurn(id, {
          status: 'error',
          errorMessage:
            err instanceof ApiError ? err.message : '요청 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
        });
      }
      return;
    }

    const skipIntentCheck = requestQuery !== (baseQuery ?? requestQuery);

    try {
      // AI 상세검색(2026-08-12) - "음료수"처럼 짧고 애매한 검색어면 다나와 실측
      // 가격 스트림을 바로 태우기 전에 먼저 물어본다. looksAmbiguous()가
      // 대부분의(구체적인) 검색어를 걸러내므로 이 호출 자체가 거의 항상 스킵된다.
      // 실패해도(.catch) 조용히 원래 검색으로 넘어간다 - AI 상세검색은 있으면
      // 좋은 보조 기능이지 필수 경로가 아니다. 후속 턴(skipIntentCheck)에서는
      // 이미 한 축을 답했으므로 다시 묻지 않는다.
      if (!skipIntentCheck && looksAmbiguous(requestQuery)) {
        const persona = { ...sessionPreferences, ...personaOverride };
        const clarify = await checkClarifyFacets(requestQuery, baseQuery, persona, getStoredToken()).catch(
          () => null
        );
        if (clarify && clarify.options.facets.length > 0) {
          patchTurn(id, { status: 'result', result: clarify });
          return;
        }
      }

      let finalResult: DecideResult | null = null;
      let streamError: string | null = null;
      await decideStream(
        requestQuery,
        (event) => {
          if (event.type === 'status') {
            patchTurn(id, { streamingStage: event.stage });
          } else if (event.type === 'proposal') {
            appendStreamingProposal(id, event.proposal);
          } else if (event.type === 'final') {
            finalResult = event.result;
          } else if (event.type === 'error') {
            streamError = event.message;
          }
        },
        undefined,
        undefined,
        skipIntentCheck
      );

      if (streamError || !finalResult) {
        throw new ApiError(streamError || '요청 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.');
      }
      patchTurn(id, { status: 'result', result: finalResult });
      persistHistoryEntry(requestQuery, finalResult).catch(() => {});
    } catch (err) {
      patchTurn(id, {
        status: 'error',
        errorMessage:
          err instanceof ApiError ? err.message : '요청 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
      });
    }
  };

  // 활성 대화가 있으면 그 대화에 이어붙이고(ChatGPT에서 입력창에 타이핑하는 것과
  // 동일), 없으면(방금 새 검색을 눌렀거나 첫 방문) 새 대화를 연다.
  const sendMessage = async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    const turn = newTurn(trimmed, trimmed);
    appendTurn(activeConversationId, turn);
    await runTurn(turn.id, turn.requestQuery, undefined, turn.baseQuery);
  };

  // clarify 카드에서 브랜드를 고르면 "새 메시지를 보낸 것"처럼 같은 대화에 이어붙인다 -
  // 원래 검색어는 그대로 두고 브랜드만 골라 재조회하는 후속 질문 취급.
  const selectBrand = async (turnId: string, brand: string) => {
    const origin = findTurn(turnId);
    const conversation = conversations.find((c) => c.turns.some((t) => t.id === turnId));
    if (!origin || !conversation) return;
    rememberPreference('브랜드', brand);
    const turn = newTurn(brand, origin.requestQuery, brand);
    appendTurn(conversation.id, turn);
    await runTurn(turn.id, turn.requestQuery, turn.brand);
  };

  // AI 상세검색 카드에서 기준(facet) 옵션을 하나 고르면 원래 검색어 뒤에 덧붙여
  // 같은 대화에 새 메시지처럼 이어붙인다. brand 파라미터를 안 써서(run_brand_price가
  // 아니라) 일반 검색 경로를 그대로 타므로, 조합한 검색어가 여전히 애매하면
  // runTurn의 clarify 선체크가 다시 걸려 자연스럽게 여러 턴에 걸쳐 좁혀나갈 수 있다.
  // baseQuery는 origin에서 그대로 물려받는다(origin.requestQuery가 아니라) -
  // 드릴다운 체인 전체가 맨 처음 검색어 하나로 고정돼야 백엔드가 매번 그
  // 하나만 캐시해서 재사용할 수 있다(속도 개선, 2026-08-13).
  //
  // selected가 {라벨: 값}으로 넘어온다(사용자 페르소나, 2026-08-15) - 값
  // 배열이 아니라 라벨을 같이 받아야 어느 facet에서 이 값을 골랐는지 세션/계정
  // 페르소나에 정확히 기록할 수 있다.
  const selectFacets = async (turnId: string, selected: Record<string, string>) => {
    const origin = findTurn(turnId);
    const conversation = conversations.find((c) => c.turns.some((t) => t.id === turnId));
    const values = Object.values(selected);
    if (!origin || !conversation || values.length === 0) return;
    Object.entries(selected).forEach(([label, value]) => rememberPreference(label, value));
    const combined = values.reduce((acc, value) => dedupeAppend(acc, value), origin.requestQuery).trim();
    // 2026-08-18(사용자 리포트: "핸드폰 한다음에 샤오미 넣었는데 샤오미만 다시
    // 검색되는게 뭐하는거야 '핸드폰 샤오미' 이렇게 전에 했던것도 붙여서 넣어야지")
    // - 실제로 백엔드에 보내는 requestQuery(=combined)는 이미 이전 검색어까지
    // 합쳐져 있었지만, 말풍선에 보여주는 displayQuery는 방금 고른 값만
    // (values.join)이라 마치 이전 맥락이 사라진 것처럼 보였다. 실제 검색어와
    // 화면 표시를 일치시킨다.
    const turn = newTurn(combined, combined, undefined, origin.baseQuery);
    appendTurn(conversation.id, turn);
    await runTurn(turn.id, turn.requestQuery, undefined, turn.baseQuery, selected);
  };

  // 고정 축(제품/용량/개수) clarify 카드에서 옵션 하나를 골랐을 때 - selectFacets와
  // 같은 방식으로 검색어에 이어붙여 같은 대화의 후속 턴을 만든다(run_brand_price로
  // 단축하지 않고 일반 검색 경로를 그대로 탄다).
  const selectClarifyOption = async (turnId: string, step: Exclude<ClarifyStep, 'brand'>, value: string) => {
    const origin = findTurn(turnId);
    const conversation = conversations.find((c) => c.turns.some((t) => t.id === turnId));
    if (!origin || !conversation) return;
    const personaLabel = CLARIFY_STEP_PERSONA_LABEL[step];
    const personaOverride = personaLabel ? { [personaLabel]: value } : undefined;
    if (personaLabel) rememberPreference(personaLabel, value);
    const combined = dedupeAppend(origin.requestQuery, value).trim();
    // 2026-08-18: selectFacets와 같은 이유로 displayQuery도 combined로 맞춘다 -
    // 방금 고른 값만 보여주면 이전 검색어가 빠진 것처럼 보인다.
    const turn = newTurn(combined, combined, undefined, origin.baseQuery);
    appendTurn(conversation.id, turn);
    await runTurn(turn.id, turn.requestQuery, undefined, turn.baseQuery, personaOverride);
  };

  const retryTurn = async (turnId: string) => {
    const turn = findTurn(turnId);
    if (!turn) return;
    patchTurn(turnId, { status: 'loading', errorMessage: '', streamingStage: null, streamingProposals: [] });
    await runTurn(turnId, turn.requestQuery, turn.brand, turn.baseQuery);
  };

  // 내 메시지 편집(사용자 요청, "클로드 너처럼 ... 편집기능") - 클로드처럼 편집한
  // 턴 이후에 이어지던 턴들은 그 편집 전 맥락으로 답한 것이라 더 이상 유효하지
  // 않으므로 버리고, 편집한 턴을 새 루트 질문 취급해 처음부터 다시 실행한다.
  // id는 그대로 유지해 리스트에서 자리가 안 바뀌게 한다.
  const editTurn = async (turnId: string, newQuery: string) => {
    const trimmed = newQuery.trim();
    if (!trimmed) return;
    const conversation = conversations.find((c) => c.turns.some((t) => t.id === turnId));
    if (!conversation) return;
    const index = conversation.turns.findIndex((t) => t.id === turnId);
    if (index === -1) return;
    const edited: ChatTurn = { ...newTurn(trimmed, trimmed), id: turnId };
    setConversations((prev) =>
      prev.map((c) =>
        c.id === conversation.id
          ? { ...c, turns: [...c.turns.slice(0, index), edited], updatedAt: Date.now() }
          : c
      )
    );
    await runTurn(edited.id, edited.requestQuery, undefined, edited.baseQuery);
  };

  const handleImageUpload = async (file: File) => {
    setOcrBusy(true);
    try {
      const { ocr, cleaned } = await extractOcr(file);

      if (cleaned?.search_query?.trim()) {
        await sendMessage(cleaned.search_query.trim());
        return;
      }

      // Groq 정제가 정상적으로 돌았는데도(에러 없음) search_query가 빈
      // 문자열이면, LLM이 "상품을 하나로 특정할 수 없다"고 일부러 포기한
      // 것이다(cleanup.py 프롬프트: "상품을 특정할 수 없으면 search_query를
      // 빈 문자열로 두세요" - 지어내지 말라는 지시). 이럴 때 다음 폴백인
      // cleaned_text로 넘어가면, 정리는 됐지만 "상품 하나로 좁히기"는 안 된
      // 텍스트(사진에 같이 찍힌 다른 물건 글자까지 포함될 수 있음)가 그대로
      // 검색어가 되어버린다(사용자 리포트, 2026-08-18: 캔 사진에 옆에 있던
      // 노트북 스티커·젤리 봉지 글자까지 검색어에 섞여 나옴). LLM의 "모르겠다"
      // 판단을 존중해 지저분한 텍스트로 검색하는 대신 정직하게 다시 찍어달라고
      // 안내한다 - API 호출 자체가 실패한 경우(cleaned.error 있음)는 여기
      // 해당 안 되고 아래 폴백 체인으로 그대로 넘어간다.
      if (cleaned && !cleaned.error) {
        appendTurn(activeConversationId, {
          ...newTurn('(이미지)', ''),
          status: 'error',
          errorMessage: '사진에서 특정 상품을 찾기 어려워요. 상품만 잘 보이게 다시 찍어주시겠어요?',
        });
        return;
      }

      // 여기부터는 Groq 정제 호출 자체가 실패한 경우(cleaned가 null이거나
      // error가 있음) - cleanup.py가 재시도 후에도 실패하면 이미 로컬 규칙
      // 필터를 거친 cleaned_text를 주므로, 그마저 없을 때만 원본 ocr.text로
      // 폴백한다.
      const extractedText = (cleaned?.cleaned_text || ocr.text || '').trim();
      if (!extractedText) {
        appendTurn(activeConversationId, {
          ...newTurn('(이미지)', ''),
          status: 'error',
          errorMessage: ocr.error || '이미지에서 텍스트를 찾지 못했습니다.',
        });
        return;
      }
      await sendMessage(extractedText);
    } catch (err) {
      appendTurn(activeConversationId, {
        ...newTurn('(이미지)', ''),
        status: 'error',
        errorMessage:
          err instanceof ApiError ? err.message : '이미지 분석 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
      });
    } finally {
      setOcrBusy(false);
    }
  };

  // "새 검색"(사이드바) - ChatGPT의 "+New chat"과 동일하게, 지금 대화는 목록에
  // 그대로 남겨두고 활성 대화만 비운다. 다음 sendMessage가 activeConversationId
  // 없음을 보고 자동으로 새 대화를 연다(appendTurn 참고) - 여기서 바로 새
  // Conversation을 만들지 않는 이유는, 사용자가 "새 검색"만 누르고 아무것도
  // 입력하지 않으면 빈 대화가 목록에 쌓이는 걸 막기 위해서다.
  const handleReset = () => {
    setActiveConversationId(null);
  };

  const loadFromHistory = (entry: HistoryEntry) => {
    const turn: ChatTurn = {
      ...newTurn(entry.query, entry.query),
      status: 'result',
      result: entry.result,
      createdAt: entry.timestamp,
    };
    const id = crypto.randomUUID();
    setConversations((prev) => [{ id, title: entry.query, turns: [turn], updatedAt: Date.now() }, ...prev]);
    setActiveConversationId(id);
  };

  const deleteFromHistory = (id: string) => {
    if (user) {
      const token = getStoredToken();
      if (token) deleteServerHistoryEntry(token, id).catch(() => {});
      setHistory((prev) => prev.filter((h) => h.id !== id));
      return;
    }
    setHistory(deleteHistoryEntry(id));
  };

  const clearAllHistory = () => {
    if (user) {
      const token = getStoredToken();
      if (token) clearServerHistory(token).catch(() => {});
      setHistory([]);
      return;
    }
    setHistory(clearHistory());
  };

  const isBusy = ocrBusy || turns.some((t) => t.status === 'loading');

  return (
    <SearchContext.Provider
      value={{
        turns,
        isBusy,
        ocrBusy,
        history,
        sessionPreferences,
        rememberPreference,
        sendMessage,
        selectBrand,
        selectFacets,
        selectClarifyOption,
        retryTurn,
        editTurn,
        handleImageUpload,
        handleReset,
        loadFromHistory,
        deleteFromHistory,
        clearAllHistory,
      }}
    >
      {children}
    </SearchContext.Provider>
  );
};

export const useSearch = () => {
  const ctx = useContext(SearchContext);
  if (!ctx) throw new Error('useSearch는 SearchProvider 내부에서만 사용할 수 있습니다.');
  return ctx;
};
