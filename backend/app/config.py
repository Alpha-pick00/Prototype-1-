import os
import secrets

from dotenv import load_dotenv

load_dotenv()


class Settings:
    qwen_api_key: str | None = os.environ.get("QWEN_API_KEY")
    # DashScope는 리전마다 별도 엔드포인트/계정이다 - 이전에 이 프로젝트가 Qwen을
    # 붙였다가 "Model Studio 계정의 과금 플랜 활성화 문제"로 포기한 적이 있는데
    # (agents/deepseek.py 주석 참고), Model Studio는 국제(비중국 본토) DashScope의
    # 제품명이라 기본값을 국제 엔드포인트로 둔다. 중국 본토 계정이면 .env의
    # QWEN_API_BASE를 https://dashscope.aliyuncs.com/compatible-mode/v1 로 바꿀 것.
    qwen_api_base: str = os.environ.get(
        "QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    deepseek_api_key: str | None = os.environ.get("DEEPSEEK_API_KEY")
    tavily_api_key: str | None = os.environ.get("TAVILY_API_KEY")
    # 11번가 오픈 API(openapi.11st.co.kr) 키 - 아직 어디서도 안 쓰인다(등록만
    # 해둠, 2026-08-20). 검색 도메인을 다나와 외로 확장할 때 쓸 수 있다.
    elevenst_api_key: str | None = os.environ.get("ELEVENST_API_KEY")

    # 2026-08-18("qwen 3.7 + 로 모델 바꿔줘") - qwen-max에서 Qwen3.7 세대의
    # plus 등급으로 교체. 필요하면 .env의 QWEN_MODEL로 다른 버전(예:
    # qwen3.7-max)으로 바꿀 수 있다.
    qwen_model: str = os.environ.get("QWEN_MODEL", "qwen3.7-plus")
    deepseek_model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # "groq"/judge 슬롯은 2026-08-16부터 Groq(무료 API)이 담당한다(사용자 요청:
    # "deepseek Qwen 빼고 싹 다 무료 모델로 바꾸려고 해" - Gemini는 프로젝트가
    # 403으로 막혀있었고 Claude는 애초에 상시 무료 티어가 없다). Groq도 OpenAI
    # 호환 엔드포인트라 gpt.py/deepseek.py와 같은 패턴(AsyncOpenAI+base_url)을
    # 그대로 쓴다. agent 식별자는 원래 "gemini"였지만 2026-08-18("Gemini
    # 이제 안쓰니까 이름 제대로 바꿔서 코드 반영해") 실제 쓰는 모델명을 따라
    # "groq"로 리네임했다(gpt 슬롯과 달리 - Qwen으로 바뀐 뒤에도 "gpt" 식별자를
    # 유지한 건 리네임 비용이 훨씬 컸기 때문).
    groq_api_key: str | None = os.environ.get("GROQ_API_KEY")
    groq_api_base: str = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    # OCR 텍스트 정리/propose의 "groq" 슬롯/대량구매(bulk) propose·심사가 공통으로
    # 쓰는 범용 모델(카테고리분류는 2026-08-19부터 groq_refine_model로 옮겼다 -
    # 바로 아래 groq_refine_model 주석 참고). 원래는 Groq 무료(on-demand) 티어의
    # 분당 토큰(TPM) 한도가 가장 넉넉한
    # llama-3.3-70b-versatile을 썼는데(검색 결과 12건을 그대로 프롬프트에 넣으면
    # 이 한도를 매번 초과했다 - agents/base.py의 _SNIPPET_MAX_CHARS로 기본 해결),
    # 2026-08-18 llama-3.3-70b-versatile이 Groq에서 완전히 내려가(계정
    # /v1/models 조회에도 안 잡힘) 모든 호출이 404로 죽는 게 확인됐고("llama
    # 모델 전부 GPT-oss 무료 모델로 바꿔줘") - 그라운딩 회귀 파일럿에서
    # gpt-oss-20b(refine과 같은 모델)로 한 번 바꿨는데, 그러자 refine + propose +
    # 카테고리분류 + OCR이 전부 gpt-oss-20b의 같은 20만 토큰/일 예산을 나눠 쓰게
    # 돼 오히려 더 빨리(같은 파일럿의 43번째 케이스 근처) 소진됐다 - 바로 아래
    # groq_refine_model과 겹치지 않게 gpt-oss-120b(judge와 공유)로 옮겼다.
    # propose는 output_schema를 안 쓰므로(순수 JSON 배열 텍스트를 직접 파싱)
    # 애초에 구조화 출력 지원 여부와 무관하게 아무 모델이나 쓸 수 있다 - judge와
    # 공유하는 이 조합이 refine과 공유하는 것보다 실제로 더 나은지는 다음
    # 파일럿으로 다시 확인해야 한다(judge는 후보가 1개면 스킵되지만 propose는
    # 매 질의 항상 실행돼, 어느 쪽이 덜 부딪히는지는 아직 추정일 뿐 실측하지
    # 않았다).
    groq_model: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    # refine은 프롬프트가 원본 질의 하나뿐이라 작지만, ADK가 output_schema를
    # response_format=json_schema로 요청한다 - groq/compound-mini는 이를 지원하지
    # 않는다("This model does not support response format json_schema"). 구조화
    # 출력을 지원하는 gpt-oss 계열 중 작은 쪽을 refine 전용으로 따로 둔다.
    #
    # 카테고리분류(app/category.py::classify_category)도 2026-08-19부터 이
    # 모델을 같이 쓴다(원래 groq_model=120b) - 스타일 가이드 게이트 때문에
    # 검색마다 무조건 한 번씩 불려서 사실상 매 요청 실행되는데, 120b 쪽에는
    # 이미 propose·judge·style_guide가 몰려있어 이 계정의 Groq 일일 한도
    # (모델별 200,000 토큰)를 가장 먼저 소진했다(실측 2026-08-19: 199,917/
    # 200,000). 분류는 16개 중 하나 + 불리언 하나 고르는 단순 작업이라 120b급
    # 추론이 필요하지 않다고 보고 옮겼다 - 예전에 propose+분류+OCR "셋을
    # 한꺼번에" 이 모델로 옮겼다가 오히려 이 모델이 더 빨리 고갈된 적이 있지만
    # (바로 위 groq_model 주석 참고), 이번엔 분류 하나만 옮겨 두 모델의 부하를
    # 더 고르게 나누는 것뿐이라 같은 문제가 재현될 가능성은 낮다고 판단했다.
    groq_refine_model: str = os.environ.get("GROQ_REFINE_MODEL", "openai/gpt-oss-20b")
    # judge(최종 심사)도 output_schema가 필요해 같은 gpt-oss 계열이지만, propose
    # 쪽보다 큰 120b를 따로 써서 최소한의 판단력 격차를 둔다.
    groq_judge_model: str = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")

    google_vision_api_key: str | None = os.environ.get("GOOGLE_VISION_API_KEY")

    # 소셜 로그인 (Google Client ID는 프론트엔드 VITE_GOOGLE_CLIENT_ID로만 쓰임 —
    # access_token으로 유저 정보를 조회하는 방식이라 백엔드는 client id가 필요 없다)
    kakao_client_id: str | None = os.environ.get("KAKAO_CLIENT_ID")
    kakao_client_secret: str | None = os.environ.get("KAKAO_CLIENT_SECRET")
    naver_client_id: str | None = os.environ.get("NAVER_CLIENT_ID")
    naver_client_secret: str | None = os.environ.get("NAVER_CLIENT_SECRET")

    # 세션(JWT) 서명 키. 지정하지 않으면 프로세스 시작 시 무작위로 생성되는데,
    # 이 경우 서버가 재시작될 때마다 기존 로그인 세션이 전부 무효화된다.
    # 실제 배포 시에는 반드시 .env에 고정값을 넣을 것 (예: python -c "import secrets; print(secrets.token_hex(32))").
    session_secret_key: str = os.environ.get("SESSION_SECRET_KEY") or secrets.token_hex(32)


settings = Settings()
