"""Qwen(DashScope) text-embedding-v3 기반 임베딩 유틸 - run_elevenst_only_debate의
"관련 상품" 랭킹과 향후 시맨틱 캐시가 공유해서 쓴다. 실측(2026-08-20)으로
text-embedding-v3(1024차원)만 이 계정 플랜에서 접근 가능함을 확인했다 -
text-embedding-v1/v2는 각각 404/AccessDenied로 막혀 있다."""

from __future__ import annotations

import math

from openai import AsyncOpenAI

from .config import settings

EMBEDDING_MODEL = "text-embedding-v3"


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_api_base, max_retries=0)


async def embed(texts: list[str]) -> list[list[float]] | None:
    """실패(키 없음·API 오류)하면 None을 돌려준다 - 호출부는 임베딩 없이
    원래 순서를 그대로 쓰는 폴백을 갖는다."""
    if not texts or not settings.qwen_api_key:
        return None
    try:
        client = _client()
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in response.data]
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
