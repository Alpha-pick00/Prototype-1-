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
    # 11번가 오픈 API(openapi.11st.co.kr) 키 - 메인 검색 흐름(app.debate.
    # run_elevenst_only_debate)과 AI 상세검색(check_clarify_facets)이 쓴다.
    elevenst_api_key: str | None = os.environ.get("ELEVENST_API_KEY")

    # 2026-08-18("qwen 3.7 + 로 모델 바꿔줘") - qwen-max에서 Qwen3.7 세대의
    # plus 등급으로 교체. 필요하면 .env의 QWEN_MODEL로 다른 버전(예:
    # qwen3.7-max)으로 바꿀 수 있다.
    qwen_model: str = os.environ.get("QWEN_MODEL", "qwen3.7-plus")
    deepseek_model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # OCR 텍스트 정리(app/ocr/cleanup.py) 전용 - console.groq.com 무료 API 키.
    groq_api_key: str | None = os.environ.get("GROQ_API_KEY")
    groq_api_base: str = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    groq_model: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

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
