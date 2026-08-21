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
    # 2026-08-20("11번가 api를 구해서 다나와를 폐기하고 11번가 쪽으로 방향을
    # 틀려고") - app.search.search()의 기본 검색 백엔드 및
    # adk_pipeline._ElevenstFetchNode의 자격증명. 미설정이면 두 곳 모두
    # 조용히 빈 결과로 스킵한다(ocr/google_vision.py와 동일한 "키 없으면
    # 스킵" 패턴) - 서버가 죽지 않는다.
    elevenst_api_key: str | None = os.environ.get("ELEVENST_API_KEY")

    # 2026-08-18("qwen 3.7 + 로 모델 바꿔줘") - qwen-max에서 Qwen3.7 세대의
    # plus 등급으로 교체. 필요하면 .env의 QWEN_MODEL로 다른 버전(예:
    # qwen3.7-max)으로 바꿀 수 있다.
    qwen_model: str = os.environ.get("QWEN_MODEL", "qwen3.7-plus")
    deepseek_model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # "groq"/judge 슬롯은 2026-08-16부터 Groq(무료 API)이 담당했었다(사용자 요청:
    # "deepseek Qwen 빼고 싹 다 무료 모델로 바꾸려고 해" - Gemini는 프로젝트가
    # 403으로 막혀있었고 Claude는 애초에 상시 무료 티어가 없다). agent 식별자는
    # 원래 "gemini"였지만 2026-08-18("Gemini 이제 안쓰니까 이름 제대로 바꿔서
    # 코드 반영해") 실제 쓰는 모델명을 따라 "groq"로 리네임했다(gpt 슬롯과 달리
    # - Qwen으로 바뀐 뒤에도 "gpt" 식별자를 유지한 건 리네임 비용이 훨씬 컸기
    # 때문).
    #
    # 2026-08-21("기업에서 hcx api 를 제공을 해줘가지고 지금 있는 llm api들
    # 일단 hcx로 바꿔줘") - 실제 호출 대상을 Naver Cloud CLOVA Studio(HCX)로
    # 다시 교체했다. "groq" 식별자/설정 이름은 이번에도 그대로 유지한다(gemini→
    # groq 리네임과 달리 이번엔 이름 자체를 바꿔달라는 요청이 없었고, 리네임
    # 비용 대비 이득이 낮다). HCX도 OpenAI 호환 엔드포인트를 제공해서
    # (https://api.ncloud-docs.com/docs/en/clovastudio-openaicompatibility)
    # gpt.py/deepseek.py와 같은 패턴(AsyncOpenAI+base_url)을 그대로 쓴다 - 인증도
    # 별도 헤더 없이 api_key 하나(Bearer 토큰)면 된다. 원래 GROQ_API_KEY/
    # GROQ_API_BASE 대신 새 HCX_API_KEY/HCX_API_BASE 환경변수를 쓴다 - 실제로
    # 가리키는 서비스가 바뀌었으니 자격증명도 새 이름으로 분리하는 편이
    # 이전 gemini→groq 전환 때와 같은 관례다(그때도 GEMINI_API_KEY를 그대로
    # 재사용하지 않고 GROQ_API_KEY를 새로 도입했다). DeepSeek(교차 검증)·
    # Qwen(제안 폴백 중 일부)은 이번 1차 전환 범위 밖 - 서로 다른 제공자를
    # 유지해 교차 검증의 독립성을 지킨다.
    groq_api_key: str | None = os.environ.get("HCX_API_KEY")
    groq_api_base: str = os.environ.get("HCX_API_BASE", "https://clovastudio.stream.ntruss.com/v1/openai")
    # OCR 텍스트 정리/propose의 "groq" 슬롯/대량구매(bulk) propose·심사, refine,
    # judge, 카테고리분류가 공통으로 쓰는 모델. Groq 시절엔 llama-3.3-70b-versatile
    # → gpt-oss-20b/120b로 몇 차례 갈아탔고(TPM/일일 토큰 한도 소진 문제로 refine·
    # judge를 서로 다른 크기로 분리해뒀었다 - 그 배경은 git 히스토리 참고), 2026-08-21
    # HCX로 전환하면서 세 슬롯(groq_model/groq_refine_model/groq_judge_model)이
    # 전부 "HCX-005"로 합쳐졌다 - HCX가 현재 OpenAI 호환 엔드포인트로 노출하는
    # 채팅 모델이 이것 하나뿐이라(https://api.ncloud-docs.com/docs/en/clovastudio-openaicompatibility),
    # Groq처럼 크기별로 나눠 쓸 선택지가 아직 없다. 세 환경변수를 각각 다른
    # 모델명으로 재정의할 수 있는 구조는 그대로 남겨뒀다 - HCX가 다른 크기의
    # 모델을 추가로 열어주면 다시 나눌 수 있다.
    groq_model: str = os.environ.get("GROQ_MODEL", "HCX-005")
    groq_refine_model: str = os.environ.get("GROQ_REFINE_MODEL", "HCX-005")
    groq_judge_model: str = os.environ.get("GROQ_JUDGE_MODEL", "HCX-005")

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
