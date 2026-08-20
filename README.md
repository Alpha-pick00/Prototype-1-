# αlpha Pick

alpha-pick-jet.vercel.app
---

## 1️⃣ 프로젝트 개요

### 프로젝트명 및 한 줄 소개

**αlpha Pick** — 하나의 검색어를 여러 AI 에이전트가 각자 조사해 제안하고, 별도의 심사 에이전트가 근거를 비교해 하나의 답으로 압축해주는 멀티에이전트 쇼핑 가격비교 서비스.

### 프로젝트 개요도

> 2026-08 통합 병합 이후 구조. 프론트는 GPT가 실시간으로 응답을 생성하는 대화형 멀티턴
> UI(`ChatTurn`)로, 백엔드는 ADK 기반 멀티에이전트 오케스트레이션과 다나와 실측 가격
> 연동을 함께 갖췄다. Human-in-the-loop 백엔드 추출 로직은 facet 기반 파이프라인
> 하나로 통합돼 있다(`/decide/clarify`·ADK 내부 안전망 공유). 그라운딩은 다나와
> 실측가 + 쿠팡 교차 확인 신호로 이중화돼 있다. 자세한 배경은
> [주요 의사결정 사항](#주요-의사결정-사항) 참고.

```mermaid
flowchart LR
    subgraph FE["Frontend · GitHub Pages"]
        GCI["GradientChatInput<br/>(대화형 입력, 사운드/애니메이션)"]
        CTX["SearchContext.runTurn<br/>(턴 · 히스토리 · baseQuery 관리)"]
        SB["사이드바<br/>(기록 · 로그인)"]
    end

    subgraph BE["Backend · FastAPI (AWS)"]
        DECIDE["POST /decide/stream<br/>(AI 오케스트레이션)"]
        CLARIFYF["POST /decide/clarify<br/>(AI 상세검색 · facet)"]
        DANAWAONLY["POST /decide/danawa-only[/stream]<br/>(LLM 미사용 실험 경로)"]
        CHAT["POST /clarify/ask<br/>(대화형 봇 질문 생성)"]
        OCR["POST /ocr/extract"]
        AUTH["/auth/*"]
        HIST["/history"]
        AC["/autocomplete"]
    end

    subgraph PIPE["AI 오케스트레이션 · Google ADK (adk_pipeline)"]
        REFINE["질의 정제<br/>(Groq)"]
        SEARCH["검색<br/>(Tavily + 검색 캐시)"]
        CAT["카테고리 분류<br/>(Groq, 16종)"]
        subgraph PROPOSE["제안 · 병렬 실행 (모델별 최선 1개)"]
            GPT["Qwen"]
            GEMINI["Groq"]
            DEEPSEEK["DeepSeek"]
            DANAWAP["다나와 실측가<br/>(A등급 최저가)"]
            COUPANGP["쿠팡 교차확인<br/>(후보 아님 · 참고 신호만)"]
        end
        MERGE["병합 · 중복 제거<br/>(최저가 매물 기준 통합)"]
        CHALLENGE["교차 검증<br/>(DeepSeek)"]
        JUDGE["최종 심사<br/>(Groq)"]
    end

    subgraph DANAWA["다나와 실측 가격 연동"]
        DSEARCH["다나와 직접 검색<br/>(search.danawa.com)"]
        PTABLE["가격표 페치 · A등급 판정<br/>(price_table.py)"]
        BRIDGE["최저가 브릿지 URL 해석<br/>(내부 AJAX 엔드포인트)"]
    end

    subgraph EXT["외부 서비스"]
        TAVILY["Tavily 검색 API<br/>(다나와 한정)"]
        VISION["Google Vision OCR"]
        OAUTH["Google · Kakao · Naver"]
    end

    DB[(SQLite)]

    GCI --> CTX
    CTX --> DECIDE
    CTX --> CLARIFYF
    CTX -- "LLM 키 없음(로컬 실험)" --> DANAWAONLY
    GCI --> CHAT
    GCI --> OCR
    SB --> AUTH
    SB --> HIST

    DECIDE --> REFINE --> SEARCH --> CAT
    SEARCH --> TAVILY
    CAT -- "축 관련성 판정<br/>(용량/개수는 카테고리별로 다름)" --> SEARCH
    SEARCH -- "브랜드/제품/용량/개수 모호<br/>(skip_clarify 없으면)" --> DECIDE
    SEARCH --> PROPOSE
    GPT & GEMINI & DEEPSEEK & DANAWAP --> MERGE --> CHALLENGE --> JUDGE
    COUPANGP -.->|참고 신호| CHALLENGE
    JUDGE -- 최종 추천 --> DECIDE

    CLARIFYF --> DSEARCH --> DANAWA
    DANAWAONLY --> PTABLE
    JUDGE -.-> BRIDGE
    DECIDE -.-> BRIDGE

    OCR --> VISION
    AUTH --> OAUTH
    HIST --> DB
    AC --> DB
```

### 적용 기술 스택

| 영역 | 스택 |
| --- | --- |
| Frontend | React 18, Vite 6, TypeScript, Tailwind CSS v4, Framer Motion(`motion`), React Router (HashRouter) |
| Backend | FastAPI, Python, httpx, PyJWT |
| 멀티에이전트 오케스트레이션 | Google ADK(`SequentialAgent`/`ParallelAgent`), LiteLLM |
| AI / 제안 · 검증 · 심사 | Qwen(DashScope) · Groq(GPT-OSS) · DeepSeek — 병렬 제안(모델별 최선 1개) / DeepSeek — 교차 검증(challenge) / Groq(GPT-OSS) — 최종 심사(judge) |
| 검색 | Tavily Search API (다나와로 도메인 한정) + 정규화 질의 기반 검색 캐시 |
| 다나와 실측 가격 연동 | 다나와 직접 검색/상세페이지 페치(`httpx` + `BeautifulSoup4`/`lxml`), 내부 AJAX 엔드포인트를 통한 최저가 판매처 브릿지 URL 해석 |
| Human-in-the-loop | DeepSeek가 상품명 목록에서 facet(라벨 자유, 상호 교차 필터링)을 추출 — `/decide/clarify`(다나와 직접 검색)와 ADK 파이프라인 내부 안전망(Tavily 결과) 두 진입점이 하나의 공유 추출 파이프라인을 씀. 되묻는 질문 문장은 Qwen이 실시간 생성(`/clarify/ask`) |
| 이미지 인식 | Google Cloud Vision (텍스트 추출) → Groq (정제 · 검색어 추출) |
| 인증 | Google / Kakao / Naver OAuth2 + JWT 기반 세션 |
| 저장소 | SQLite (검색 기록 · 자동완성 인덱스 · 검색 캐시) |
| 배포 | Docker, nginx, certbot, AWS GPU 인스턴스, nip.io(Backend) / GitHub Pages, Vercel(Frontend), GitHub Actions(CI) |

### 주제 선정 배경

쇼핑을 위해 여러 플랫폼 탭을 오가며 가격을 직접 비교해야 하는 번거로움에서 출발했다. 단순히 최저가를 나열하는 비교 서비스가 아니라, "왜 이 상품인지" 근거를 함께 제시하는 서비스를 목표로 했고, 하나의 LLM에만 의존할 경우 생기는 편향·환각 문제를 줄이기 위해 **여러 모델이 각자 조사해 제안하고, 별도 모델이 심사하는 멀티에이전트 구조**를 채택했다.

### 목표 및 기대효과

- 여러 쇼핑몰을 직접 비교하는 시간을 줄이고, 근거가 붙은 단일 추천으로 의사결정을 단순화
- 단일 모델 호출 대비, 여러 모델의 교차 검증을 통해 추천의 신뢰도를 높임
- 텍스트뿐 아니라 상품 사진(OCR)으로도 검색이 가능해 입력 장벽을 낮춤

### 팀원 구성 및 역할 분담

| 팀원 | 주요 역할 |
| --- | --- |
| parkminsung45 | 백엔드 멀티에이전트 토론 엔진, 검색 품질(Tavily 연동/필터링), 소셜 로그인, 배포(AWS/Docker/nginx), 프론트엔드 UI/UX 전반 |
| tmdals3000 | 검색어 자동완성(cold-start) 기능, 멀티턴 대화 기능 |
| lou0-ux | OCR 텍스트 추출 파이프라인(Google Vision + Groq 정제) |
| Seojeong Woo | 서버 인스턴스 관리 , 데이터베이스 구축, 리서치 |

### 시간순 변경 이력

날짜는 실제 커밋 기준(`git log`). 아래는 그날의 핵심만 압축한 타임라인이고,
"왜 그렇게 했는지"의 근거는 바로 아래 [주요 의사결정 사항](#주요-의사결정-사항)과
[문제 해결 내역](#문제-해결-내역-troubleshooting)에 항목별로 자세히 남아있다.

| 날짜 | 도입 / 변경 / 개선 |
| --- | --- |
| 2026-08-04 ~ 06 | Figma Make로 뽑은 포트폴리오 템플릿(Cherry-Pick)을 실 프로젝트 구조로 전환(→ Étiquette 리브랜드). FastAPI 백엔드 스캐폴딩 + GPT·Gemini·DeepSeek 멀티에이전트 구매 의사결정 엔진 최초 구현 |
| 2026-08-07 | Google·Kakao·Naver 소셜 로그인, 계정별 검색 기록 사이드바, OCR(Google Vision + Gemini 정제) 이미지 검색 파이프라인 추가. 검색 소스를 네이버쇼핑 → Google Merchant Center로 교체. AWS GPU 인스턴스 + nip.io 기반 배포 최초 구축 |
| 2026-08-08 ~ 09 | "How We Curate"(멀티에이전트 토론 흐름 설명) 섹션, README 프로젝트 리포트 섹션 신설 |
| 2026-08-10 | 다나와 실측 가격 어댑터 최초 구현(판매처별 가격표 파싱, STEP 1~5 라이브 검증) · 쿼리 정규화 검색 캐시 도입 · `fusion.dedup` 후보 병합 가드(가격 호환성 + 이름 유사도) 추가 · Étiquette → αlpha Pick 리브랜드 |
| 2026-08-11 | 다나와 A등급(구매 링크 생성 가능) 후보를 judge 풀에 직접 승격(PART 4-2) · 동일 상품 판정 기준을 판매처+가격 → 상품명으로 전환(STEP 6) · **Google ADK 기반 역할 분리 멀티에이전트 파이프라인 + 의미 기반 검색 캐시 도입(현재 아키텍처의 골격)** · Human-in-the-loop 최초 도입 |
| 2026-08-12 | 카테고리 기반 HITL 축 최적화(Gemini 16종 분류로 용량/개수 관련성 판정) · 다나와 최저가 URL 해석(브릿지 엔드포인트) + 대화형 HITL(LLM이 되묻는 문장 생성) 추가 |
| 2026-08-13 | "gpt" 에이전트 슬롯을 OpenAI → Qwen(DashScope)으로 전환 · 완전 무관 후보뿐일 때의 relaxed fallback 최초 추가 · ChatGPT식 멀티턴 대화 스레드(`ChatTurn`)로 프론트 전환 · `skip_clarify`로 재질문 반복 버그 수정 · 죽은 코드/미사용 npm 의존성 1차 정리 |
| 2026-08-14 | Gemini·Claude → Groq 무료 API 전면 전환 · `/decide/clarify`와 ADK 내부 안전망의 facet 추출 로직 통합 · "용기형태" facet 구매유형 오분류 수정 · 쿠팡 교차 확인(challenge 3번째 그라운딩 신호) 추가 · 깨진 쿠팡 구매링크 노출 버그 3건(연쇄 원인) 수정 · 액세서리(핸드폰 케이스 등) 검색 품질 개선 · 대규모 죽은 코드 정리 · README 대폭 갱신 |
| 2026-08-16 | relaxed fallback을 challenge 재검증으로 게이팅해 하드닝(그라운딩 우회 경로 차단) · `Decision.verified` 필드 추가로 최종 응답의 그라운딩 검증 여부를 API 전체에 노출 · 네이버쇼핑을 쿠팡과 동일 패턴의 2번째 소프트 교차 확인 소스로 추가 · 알려진 상품 세트 기반 그라운딩 정확도 회귀 스크립트(`scripts/grounding_regression.py`) 추가 · 다나와 실측가 후보에 검색어 관련성 가드 추가(아이폰→아이패드 오추천 버그 수정) · facet crossfilter로 이미 좁혀진 축은 되묻지 않도록 수정(불필요한 clarify 다발 버그) · 다나와 가격비교 페이지 자체를 최종 후보로 받아들이던 버그 수정 |
| 2026-08-17 | 다나와 가격비교 페이지 필터를 도메인 기반으로 일반화(모바일 URL 변형 누락 대응) · 그라운딩 회귀 스크립트에 실행 전 제공자 헬스체크 + 도중 연속 실패 시 즉시 중단 안전장치 추가 · README에 그라운딩 회귀 실험 이력을 표+그래프로 자동 갱신하는 기능 추가 · 배포 저장소를 Prototype-1- 하나로 일원화(구 Alpha-pick00.github.io가 비공개/개명되며 배포 대상에서 제외, Pages 활성화 + 누락 환경변수 설정 + 죽은 배포 터널 재기동) · 안전장치의 쿼터 소진 감지가 파이프라인 내부 예외 삼킴에 뚫리는 문제 발견 후 문자열 매칭 → 연속 실패 기반 헬스체크 재확인 방식으로 재설계 |
| 2026-08-18 | 배포 터널 재소진 + 구 GitHub Pages URL 404 확인 후 터널 재기동·`VITE_API_URL` 갱신·재배포로 복구 · "gemini" 슬롯 기본 Groq 모델을 llama-3.3-70b-versatile → gpt-oss-20b로 교체 · 프론트엔드를 Vercel에도 배포하고 백엔드를 기존 AWS 인스턴스에 최신 코드로 재배포(저장소 재동기화, nginx+TLS를 새 인스턴스 IP로 재발급), CORS에 Vercel 도메인 추가 |
| 2026-08-19 | 취향 주도 카테고리(패션의류/잡화 등)에 스타일 가이드 응답 모드 추가(검증된 후보를 스타일별로 그룹핑) · 토큰 사용량 최적화(clarify facet 추출 가드, classify_category 모델 재배정) · 저장소 전반 죽은 코드/미사용 설정·의존성 정리(백엔드·프론트엔드) · README 정리 |

### 주요 의사결정 사항

- **검색 데이터 소스**: Google Merchant API → Tavily 검색 API + 국내 리테일러 15곳 도메인 한정
- **판단 구조**: 단일 모델 호출 → ChatGPT · Gemini · DeepSeek 3개 병렬 제안 + Claude 심사의 4단계 구조
- **Google 로그인 방식**: 공식 iframe 버튼 → `google.accounts.oauth2` 팝업 + 커스텀 버튼
- **CORS 정책**: origin을 알려진 도메인으로만 제한(와일드카드 금지)
- **검색 기록 저장**: 로그인 시 서버(SQLite), 비로그인 시 로컬(localStorage)로 분기
- **판단 구조 재설계**: 단일 호출 구조를 Google ADK 기반 정제 → 검색 → 제안(3모델 병렬) → 병합 → 교차 검증 → 심사 파이프라인으로 분리
- **후보 병합 기준**: 필드별(가격 · 판매처 · URL) 독립 다수결 → 최저가 매물 하나에서 세 필드를 함께 채택
- **Human-in-the-loop 도입 방식**: ADK 내부 pause/resume 대신 앱 레벨 무상태 재실행 채택 — 브랜드 · 제품 · 용량 · 개수가 모호하면 파이프라인을 멈추고 한 축씩 되묻는다
- **카테고리 기반 HITL 축 최적화**: 검색어를 16개 대분류로 분류해, 카테고리별로 용량 · 개수 축의 관련성을 다르게 판정(예: 식품 중 음료만 용량 유효)
- **다나와 실측 가격 직접 연동**: 다나와 검색결과/상세페이지를 직접 페치해 A등급 판매처 실측가 확보, 대조 가능하면 `price_source`를 `danawa_offer`로 표시
- **다나와 가격비교 페이지 → 구매 URL 변환**: 최종 URL이 다나와 페이지면 내부 AJAX 엔드포인트로 최저가 브릿지 URL을 조회해 치환
- **검색 도메인 15곳 → 다나와로 축소**: 리테일러마다 페이지 구조가 달라 스니펫 파싱 시 오매칭이 있어, 가격비교 사이트 하나로 좁힘(에누리는 어댑터 없이 노출만 되던 상태라 함께 제외)
- **Human-in-the-loop 이원화**: 고정 4축(GPT, Tavily 결과 기반) + AI 상세검색 facet(DeepSeek, 다나와 결과 기반) 병행 — 짧은 질의는 facet을 먼저 시도하고 못 찾으면 고정 축으로 폴백
- **대화형 UI로 통합**: 프론트를 `ChatTurn` 배열 기반 멀티턴 스레드로 재구성, 브랜드/facet/축 선택을 전부 새 턴으로 통일
- **AI 오케스트레이션과 다나와 통합 병합**: 두 갈래로 개발되던 기능을 ADK 파이프라인 하나로 병합, `skip_clarify` 플래그로 재질문 회귀 수정
- **"gpt" 슬롯을 GPT → Qwen으로 교체**: 내부 식별자(`agent="gpt"`, 파일/함수명)는 유지, 호출 모델만 DashScope Qwen으로 교체. 프론트 표시 이름만 "Qwen"으로 변경
- **Gemini · Claude → Groq로 교체**: `agent="gemini"` 식별자는 유지, 호출 모델만 교체. refine은 `gpt-oss-20b`, judge는 `gpt-oss-120b`, categorize/OCR정제/propose "gemini" 슬롯은 `llama-3.3-70b-versatile`. 검색 결과 스니펫을 500자로 잘라 담도록 `format_results_block` 조정
- **다나와 A등급 실측가 주입을 ADK 파이프라인으로 포팅**: `_DanawaFetchNode`를 propose의 `ParallelAgent` 소속으로 추가, 기존 병합/그라운딩 로직에 그대로 태움. 이전엔 `DecideResponse.price_table`이 라이브 경로에서 항상 null이었음
- **clarify 백엔드 추출 로직을 facet 하나로 통합**: `/decide/clarify`와 ADK 내부 안전망이 같은 facet 추출 파이프라인(`_extract_facets`)을 공유하도록 통합, 입력 소스만 다르게 유지. 고정 4축 전용 헬퍼는 facet 버전으로 교체하고 원본 삭제
- **facet 크로스필터를 하이퍼그래프 incidence 구조로 재구성**: 브루트포스 재스캔 방식을 `_build_facet_value_incidence`(값 → 상품 인덱스 집합) 기반 집합 연산으로 교체, 결과는 기존과 동치(테스트로 검증)
- **사용하지 않는 코드 일괄 정리**: 레거시 직접-구현 경로(`run_single_debate_price_table_variant`와 그 전용 헬퍼), 고정 4축 전용 필터 함수, 미사용 정규식 헬퍼, 미사용 병합 함수, 도달 불가능한 중복 코드, 미사용 프론트 데모 라우트/scaffold, 미사용 SSE 클라이언트/clarify-match 엔드포인트, 미사용 prop/훅을 전수 조사 후 제거
- **쿠팡 검색을 challenge 단계의 3번째 그라운딩 소스로 추가**: `search.search_coupang()`으로 독립된 쇼핑몰 신호를 얹어 `_CoupangCheckNode`가 propose와 동시 실행(지연시간 추가 없음). 페이지를 직접 파싱하지 않고 Tavily 스니펫만 challenge 참고 자료로 전달, 소프트 신호로만 사용
- **그라운딩 3종 강화**: (1) relaxed fallback도 정상 경로와 동일한 challenge 검증을 거치도록 하드닝, `Decision.verified` 필드로 검증 여부를 API 전체에 노출 (2) 네이버쇼핑을 쿠팡과 동일한 패턴의 2번째 소프트 교차 확인 소스로 추가 (3) `scripts/grounding_regression.py` 그라운딩 정확도 회귀 스크립트 추가
- **실험 안전장치 도입 및 재설계**: 제공자 쿼터 소진 감지를 "실패 텍스트의 소진 신호 문자열 매칭" 방식에서 "사유 불문 연속 2건 실패 시 헬스체크로 직접 재확인" 방식으로 재설계(파이프라인이 예외를 어떻게 감싸든 영향받지 않음). 오염된 실행 결과는 히스토리에서 되돌림
- **README 그라운딩 실험 이력 자동 갱신**: `scripts/grounding_regression_history.json`에 완주한 실행마다 결과를 append하고, README의 `GROUNDING_HISTORY_START/_END` 구간을 표 + Mermaid 그래프로 자동 재생성
- **배포 저장소를 Prototype-1- 하나로 일원화**: GitHub Pages 활성화, 배포 환경변수 설정, Cloudflare 터널 재기동
- **gemini 슬롯 기본 Groq 모델을 gpt-oss-20b로 교체**: 이후 그라운딩 파일럿에서 20b가 refine과 예산을 나눠 쓰며 더 빨리 고갈되는 게 확인돼, judge와 공유하는 `gpt-oss-120b`로 재조정
- **프론트엔드를 Vercel에도 배포하고 백엔드를 AWS 인스턴스로 이전**: 기존 Cloudflare Quick Tunnel을 벗어나 AWS EC2 인스턴스로 백엔드 이전(`backend/deploy/DEPLOY.md` 참고), nginx/TLS를 새 IP로 재발급. 프론트는 GitHub Pages를 유지한 채 Vercel에 추가 배포, CORS에 Vercel 도메인 추가
- **토큰 사용량 최적화**: `_extract_clarify_options`가 후속 질의 라운드에도 무거운 facet 추출(브랜드별 최대 15개 병렬 DeepSeek 호출)을 무조건 실행하던 것을 가드 처리, 브랜드별 팬아웃도 6개로 제한. `classify_category`를 부하가 몰린 `gpt-oss-120b`에서 여유 있는 `gpt-oss-20b`로 재배정
- **저장소 정리**: 호출부가 없는 함수/클래스, 옛 프로토타입 디렉터리, 대체된 Google Merchant/임베딩 기반 검색 캐시 모듈과 그 설정·의존성을 제거. 프론트의 미사용 멀티 대화 전환 상태, 중복 CSS 파일, 빈 PostCSS 설정 제거

### 문제 해결 내역 (Troubleshooting)

- **검색 품질 저하**: 목록/콘텐츠 페이지가 검색 결과에 섞이는 문제 → 도메인 화이트리스트 + 제네릭 목록 URL 정규식 필터링 + 브랜드-URL 일치 검증으로 수정
- **정규식 오탐**: `search.shopping.naver.com`이 제네릭 목록 URL로 오분류 → 부정 후방탐색(negative lookbehind)으로 수정
- **동일 상품 병합 시 필드 불일치**: 가격 · URL · 판매처를 필드별로 독립 다수결 처리해 서로 다른 상품의 필드가 섞임 → 최저가 매물 하나에서 세 필드를 함께 채택하도록 수정
- **Human-in-the-loop 선택이 수렴하지 않음**: 이미 답한 조건을 매 검색마다 재추출해 같은 질문을 반복 → 질의 텍스트에 이미 반영된 조건은 재추출 결과와 무관하게 확정 처리
- **자동완성 추천창이 결과 화면 뒤에 남음**: 검색 상태와 무관하게 질의 변경마다 자동완성이 재오픈 → idle 상태일 때만 노출되도록 수정
- **멀티턴 드릴다운이 수렴하지 않음**: 내부 애매함 판정이 `skip_intent_check` 플래그와 무관하게 매번 재동작해 같은 질문이 반복 → `skip_clarify` 플래그를 파이프라인 끝까지 관통시켜 후속 턴에서 조기 종료를 건너뛰도록 수정
- **"용기형태" facet에 구매유형 값이 섞임**: facet 추출 프롬프트가 "용기형태"의 의미를 정의하지 않아 구매유형 수식어를 물리적 용기 형태로 오분류 → 프롬프트에 두 라벨을 명시하고, 코드 레벨 블랙리스트 필터 추가
- **"핸드폰 케이스" 검색 품질 저하 3종**: (1)(2) 구매유형/특징 facet에 근거 없는 값이 뜸 → 라벨 정의를 프롬프트에 명시 + 화이트리스트 필터 추가 (3) 옛 모델이 검색 결과에 섞임(상품명 유사도만으로 동일 상품 판정해 다른 모델이 병합됨) → 모델/규격 토큰 충돌 가드를 `app.spec_match`로 공용화해 병합 단계에도 적용
- **깨진 쿠팡 구매링크가 최종 추천으로 노출됨**: 다나와의 쿠팡 제휴 코드(`TP40F`) 자체가 접근 제한됨 → `danawa_mall_map.py`에서 A등급 판정 제외. 연쇄로 발견된 관련 버그 2건도 함께 수정(bridge_passthrough 재확인 강화, `/bridge/` 경로를 재해석 대상에서 제외)
- **다나와 실측가 후보가 검색어와 무관한 상품을 추천함**: `pick_primary()`가 판매처 개수만으로 대표 페이지를 골라 관련성을 확인하지 않음 → 후보 생성 전에 검색어와의 이름 매칭 가드 추가
- **구체적인 검색어인데도 불필요하게 되묻기가 뜸**: `_facet_resolved`가 문자열 완전 일치만 확인해 브랜드명과 제조사명이 다르면 매칭 실패 → crossfilter 기반 판정(`_facet_options_for_query`) 추가
- **최종 추천의 판매처가 "다나와" 자신, 가격은 빈 문자열로 노출됨**: 다나와 가격비교 페이지 자체가 challenge를 통과함 → `is_danawa_comparison_page()`를 후보 필터와 relaxed fallback에 연결해 입구에서 차단
- **다나와 가격비교 페이지 필터가 모바일 URL 변형을 놓침**: 정규식이 PC 경로만 걸렀음 → 도메인 + 경로 기반 일반 판정 방식으로 교체
- **쿼터 소진 안전장치가 파이프라인의 내부 예외 삼킴에 뚫림**: 원본 429 예외가 내부에서 일반 예외로 감싸져 문자열 매칭이 무력화됨 → "사유 불문 연속 2건 실패" 트리거 + 헬스체크 재확인 방식으로 재설계, 오염된 결과는 되돌림
- **구 GitHub Pages URL이 404, 배포 API 터널이 재차 다운**: 저장소명 변경으로 Pages URL 규칙이 깨지고, 동시에 Cloudflare Quick Tunnel이 재연결 루프에 빠짐 → 새 터널 기동 + `VITE_API_URL` 갱신 + 재배포로 복구
- **새로 발급받은 Qwen 키가 기존 워크스페이스 엔드포인트에서 거부됨**: 새 키가 다른 워크스페이스 소속으로 확인 → 직전까지 정상 동작하던 키로 롤백
- **OCR 정제/카테고리분류/propose "gemini" 슬롯이 전부 404로 실패**: Groq가 `llama-3.3-70b-versatile`을 무료 티어에서 서비스 종료 → `gpt-oss-20b`로 교체. 이후 refine과 예산을 나눠 쓰며 더 빨리 소진되는 게 확인돼 `gpt-oss-120b`로 재조정
- **AWS 재배포 직후 실제 검색이 전부 실패**: Tavily가 플랜 한도 초과(432) 반환 → 새 키로 교체, 로컬/AWS 양쪽 `.env` 갱신
- **Vercel GitHub 연동 프리뷰 빌드가 매번 실패**: Root Directory 설정이 비어 있어 리포 루트에서 빌드 시도 → Vercel API로 `rootDirectory: "frontend"` 설정

---

## 2️⃣ Project 과정 기록

### 프로젝트 목표 및 배경

여러 쇼핑몰의 가격을 일일이 비교하는 수고를 없애고, 근거가 있는 단일 추천을 제공하는 것이 목표. (배경은 [1️⃣ 주제 선정 배경](#주제-선정-배경) 참고)

### 데이터 소스 및 탐색

- **검색 데이터**: Tavily Search API를 통해 실시간으로 조회, 다나와 도메인으로 한정(원래 국내 리테일러 15곳이었으나, 사이트마다 페이지 구조가 달라 스니펫만으로 파싱하면 엉뚱한 상품이 섞이는 문제로 가격비교 사이트 하나로 축소)
- **다나와 실측 데이터**: 다나와 검색결과/상세페이지를 직접 페치해 판매처별 가격 · 배송정보 · 구매 링크 가능 여부(A/B등급)를 파싱, 내부 AJAX 엔드포인트로 최저가 판매처의 실제 구매 URL까지 확보
- **이미지 데이터**: 사용자가 업로드한 상품 사진 → Google Cloud Vision으로 텍스트 추출

### 전처리(검색 결과 정제) 방법

- 상품 상세/가격 정보가 없는 콘텐츠·매거진·검색결과 목록 도메인 제외 (`EXCLUDE_DOMAINS`)
- 정규식 기반 제네릭 목록 URL 필터링 (`is_generic_listing_url`)
- 브랜드-URL 그라운딩 검증으로 무관한 상품이 섞이는 것을 방지
- OCR 원문에서 가격/바코드/프로모션 문구를 제거하고 상품명·용량 등 핵심 메타데이터만 남기는 Groq 정제 단계(`search_query` 추출)

### 평가 기준 (무엇으로 "좋은 답"을 판단할지)

- 실제 판매 중인 상품 페이지 URL인지 (목록/콘텐츠 페이지 배제)
- 검색어의 브랜드·상품과 실제 반환된 상품이 일치하는지
- 최종 추천에 가격·판매처·선정 근거가 모두 포함되는지

### 베이스라인 대비 개선

단일 LLM 호출(베이스라인) 대비, 3개 제안 모델 + 1개 심사 모델의 멀티에이전트 구조를 통해 한 모델의 편향·환각이 곧바로 최종 답이 되는 것을 방지하도록 설계했다.

### 아키텍처 (역할 분리형 에이전트 파이프라인 · Google ADK)

```mermaid
sequenceDiagram
    participant U as 사용자
    participant CTX as SearchContext.runTurn
    participant B as 백엔드(ADK 파이프라인)
    participant Cache as 검색 캐시
    participant T as Tavily
    participant P as 제안 에이전트(Qwen·Groq·DeepSeek·다나와실측)
    participant CP as 쿠팡(교차확인 · 참고신호)
    participant D as DeepSeek(교차 검증)
    participant J as Groq(심사)
    participant DW as 다나와(브릿지 URL 해석)

    U->>CTX: 검색어 입력(첫 턴)
    CTX->>B: POST /decide/stream (skip_intent_check=false)
    B->>B: 질의 정제(Groq)
    B->>Cache: 캐시 조회
    alt 캐시 미스
        B->>T: 다나와 한정 검색
        T-->>B: 검색 결과
        B->>Cache: 결과 저장
    end
    alt 브랜드/제품/용량/개수 모호 (Human-in-the-loop)
        B-->>CTX: mode: clarify (고정 축 옵션)
        CTX-->>U: 새 턴으로 이어붙여 되묻기(버튼 · 채팅 둘 다)
        U->>CTX: 옵션 선택 또는 채팅 답변(Qwen이 매칭)
        CTX->>B: 후속 턴 POST /decide/stream (skip_intent_check=true)
        Note over B: skip_clarify=true → 내부 애매함 판정을 건너뛰고<br/>바로 제안 단계로 진행(재질문 방지)
    end
    B->>P: 검색 결과 + 질의 전달 (병렬, 모델별 최선 1개)
    P-->>B: 상품 후보 제안 (근거 포함, 다나와는 실측가)
    B->>CP: 병렬로 쿠팡 한정 검색(후보 아님)
    CP-->>B: 참고용 검색 결과
    B->>B: 후보 병합 · 중복 제거(최저가 매물 기준)
    B->>D: 병합된 후보 + 쿠팡 참고 결과로 교차 검증 요청
    D-->>B: 검증 결과(verified 여부 · note)
    B->>J: 검증된 후보 심사 요청
    J-->>B: 최종 추천 + 선정 근거
    B->>DW: 최종 URL이 다나와 페이지면 최저가 브릿지 URL 조회
    DW-->>B: 실제 구매 가능 URL
    B-->>CTX: 상품명 · 가격 · 판매처 · 근거 (스트리밍)
    CTX-->>U: 대화 스레드에 결과 카드 표시
```

짧고 애매한 검색어(예: "핸드폰")는 위 흐름 전에 `POST /decide/clarify`(다나와 검색 결과 기반 동적 facet, DeepSeek)를 먼저 시도하고, facet을 못 찾으면 그대로 `/decide/stream` 경로로 넘어간다.

### 트러블슈팅

[1️⃣ 문제 해결 내역](#문제-해결-내역-troubleshooting) 참고.

### 성능/품질 개선 기록

- 검색 도메인을 다나와로 좁혀 신뢰도 낮은 결과 원천 차단(가격비교 사이트 특성상 여러 판매처를 한 페이지에서 일관된 구조로 비교 가능)
- 제네릭 목록 URL·브랜드 불일치 필터링으로 "판매 페이지로 연결되지 않는" 문제 해결
- OCR 결과를 원문 그대로 검색하지 않고 정제된 `search_query`만 사용해 검색 적중률 개선
- 동일 상품 후보 병합 시 가격 · 판매처 · URL을 최저가 매물 하나에서 함께 채택하도록 바꿔 "가격과 실제 연결 URL이 다른 상품" 불일치 제거
- 제안/교차 검증 프롬프트에 브랜드 · 제품 · 용량 · 개수 정확 일치 조건을 명시해, Human-in-the-loop으로 이미 좁힌 조건이 검색 품질 문제로 다시 섞이지 않도록 개선
- 카테고리별로 용량 · 개수 축의 관련성을 다르게 판정해(Groq 16종 분류), 해당 없는 축을 억지로 고르게 해 상품 매핑이 틀어지는 문제 감소
- AI 상세검색(facet) 다중 라운드 시 base_query를 유지해 다나와 검색 캐시(1시간, 10초 crawl-delay)를 재사용하도록 개선해 드릴다운 응답속도 단축
- 다나와 실측 최저가를 별도로 확보해 LLM 추정 가격 · URL의 오차를 줄이고, 최종 URL이 다나와 가격비교 페이지 자체로 남지 않도록 실제 구매처 브릿지 URL로 항상 변환
- 멀티턴 대화 흐름에서 후속 턴에 `skip_clarify`를 적용해, 이미 답한 조건에 대해 파이프라인이 다시 되묻는 무한 재질문을 제거

### 그라운딩 회귀 실험 기록

`scripts/grounding_regression.py`(카테고리별 50개 질의, [주요 의사결정 사항](#주요-의사결정-사항)의
"그라운딩 3종 강화" 참고)를 돌릴 때마다의 통과율 추이. "정답"은 사람이 매긴 가격/상품이 아니라
구조적 검증(실제 구매 링크인지 · 그라운딩 검증 통과 여부 · 상품명 키워드 일치)만 자동 채점한다.

<!-- GROUNDING_HISTORY_START -->
실행할 때마다 이 표/그래프가 자동으로 갱신된다(`scripts/grounding_regression.py`가
`scripts/grounding_regression_history.json`에 결과를 추가하고 이 구간을 재생성한다 -
수동으로 이 마커(`GROUNDING_HISTORY_START`/`_END`) 사이를 직접 편집하지 말 것,
다음 실행 때 덮어써진다).

| 날짜 | 통과율 | 통과/전체 | 내용 | 인프라 참고 |
| --- | --- | --- | --- | --- |
| 2026-08-16 | 34% | 17/50 | PR #21~24(그라운딩 하드닝) 적용 전 베이스라인 - 아이폰→아이패드 환각, 과다 되묻기, 구매링크 미해석 버그를 이 실행에서 처음 발견 | Groq 일일 토큰 한도가 약 36/50 지점에서 소진(1~35번은 인프라 정상, 이후는 노이즈 가능) |
| 2026-08-17 | 12% | 6/50 | PR #21~24(그라운딩 하드닝) 적용 후 재검증 - 아이폰→아이패드류 환각 재발 0건 확인, 다나와 URL 필터의 모바일 변형 누락을 새로 발견(PR #25로 수정) | Qwen(DashScope) 무료 티어가 실행 초반부터 거의 소진되어 3개 제공자 중 사실상 DeepSeek만 남음 - 통과율(12%)은 코드 품질이 아니라 인프라 상태를 반영, 참고용으로만 볼 것 |
| 2026-08-18 | 10% | 5/50 | 새 Tavily 키 교체 + gemini 슬롯 gpt-oss-20b 전환 후 재검증 | 43번째 케이스 근처부터 gpt-oss-20b 일일 토큰(TPD) 한도 소진(refine과 같은 모델을 공유해 예상보다 빨리 소진 - PR #34로 judge와 공유하는 gpt-oss-120b로 재조정) - 1~42번은 인프라 정상이라 그 구간의 과다 clarify/후보 없음 실패는 실제 파이프라인 동작을 반영, 43번 이후는 노이즈 가능 |

```mermaid
xychart-beta
    title "그라운딩 회귀 파일럿 통과율 추이(%)"
    x-axis ["2026-08-16", "2026-08-17", "2026-08-18"]
    y-axis "통과율 (%)" 0 --> 100
    bar [34, 12, 10]
    line [34, 12, 10]
```

그래프의 특정 지점이 유독 낮다고 코드가 나빠졌다는 뜻은 아닐 수 있다 -
표의 "인프라 참고" 칸에 그 실행에서 제공자 쿼터 문제가 있었는지 항상 같이 본다.
<!-- GROUNDING_HISTORY_END -->

### 코드 정리 및 GitHub 관리

- 기능 단위 브랜치 → PR → 리뷰(빌드/타입체크) → merge 워크플로를 일관되게 적용 (PR #1~#28)
- 병합 완료된 브랜치는 주기적으로 감사(merge-base 확인) 후 정리해 브랜치 목록을 최신 상태로 유지
- `.env`, SQLite 데이터 파일(`autocomplete.db`, `history.db`) 등 비밀/로컬 데이터는 `.gitignore`로 관리

### 한계점 및 향후 과제

- 카카오 로그인은 REST API 키 설정을 완료했으나, 실사용 트래픽 기준의 검증은 아직 진행 전
- 정성적 검증 위주로 진행되어, 정량적 지표(응답 정확도·지연 시간 등) 기반의 자동화된 평가 체계는 부재
- 현재는 다나와 하나로 한정된 검색 범위를 점진적으로 확장할 여지가 있음
- Google ADK가 출시 초기 버전(`SequentialAgent`/`ParallelAgent`가 이미 deprecated 표시)이라, 향후 문서가 더 풍부한 `Workflow`/`@node` API로의 이전을 검토할 필요가 있음
- Human-in-the-loop을 앱 레벨의 무상태 재실행(파이프라인을 처음부터 다시 실행)으로 구현해 단계마다 정제/검색 비용이 다시 발생함 — ADK 세션 기반의 내부 pause/resume으로 전환하면 절감 가능
- clarify의 백엔드 추출 로직은 facet(DeepSeek) 하나로 통합했지만(아래 의사결정 참고), 프론트엔드의 `FixedAxisClarifyCard`(자연어 질문 생성용 `/clarify/ask`)와 브랜드 전용 버튼 블록은 아직 별도 UI로 남아있음 — 완전한 UI 수준 수렴은 후속 과제

### 회고

> `[팀 회고 내용 추가]`
