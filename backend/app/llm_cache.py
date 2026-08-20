"""Supabase 기반 LLM 응답 캐시 - 2단.

1) KV 캐시(exact_get/exact_set) - namespace+query가 완전히 같을 때만 재사용.
2) Semantic 캐시(semantic_get/semantic_set) - Qwen 임베딩 코사인 유사도가
   임계값을 넘으면(완전히 같은 문장이 아니어도) 재사용.

SUPABASE_URL/SUPABASE_KEY가 없으면(.env 미설정) 모든 함수가 안전하게
no-op(캐시 미스처럼 동작)한다 - elevenst_api_key와 같은 패턴이라 이 모듈이
없어도 나머지 파이프라인은 그대로 동작한다. 스키마는
supabase/llm_cache.sql 참고 - 이 파일은 그 테이블/함수가 이미 존재한다고
가정한다."""

from __future__ import annotations

import hashlib
import logging

from supabase import AsyncClient, acreate_client

from . import embeddings
from .config import settings

logger = logging.getLogger(__name__)

KV_TABLE = "llm_cache"
SEMANTIC_TABLE = "llm_semantic_cache"
SEMANTIC_MATCH_FN = "match_llm_semantic_cache"
# 오탐(다른 의도의 질의인데 캐시를 잘못 재사용)이 잘못된 추천으로 바로
# 이어지므로 보수적으로 높게 둔다 - 재사용 이득보다 정확도가 우선이다.
SEMANTIC_SIMILARITY_THRESHOLD = 0.95

_client: AsyncClient | None = None
_client_checked = False


def _cache_key(namespace: str, query: str) -> str:
    return hashlib.sha256(f"{namespace}:{query}".encode()).hexdigest()


async def _get_client() -> AsyncClient | None:
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if not settings.supabase_url or not settings.supabase_key:
        return None
    try:
        _client = await acreate_client(settings.supabase_url, settings.supabase_key)
    except Exception:
        logger.exception("Supabase 클라이언트 초기화 실패 - 캐시 없이 계속 진행")
        _client = None
    return _client


async def exact_get(namespace: str, query: str) -> dict | None:
    """namespace+query가 완전히 같은 과거 호출의 캐시된 응답. 미스·오류·
    미설정이면 None."""
    client = await _get_client()
    if client is None:
        return None
    try:
        result = (
            await client.table(KV_TABLE)
            .select("response")
            .eq("cache_key", _cache_key(namespace, query))
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0]["response"] if rows else None
    except Exception:
        logger.exception("KV 캐시 조회 실패 - 캐시 미스로 처리")
        return None


async def exact_set(namespace: str, query: str, response: dict) -> None:
    client = await _get_client()
    if client is None:
        return
    try:
        await client.table(KV_TABLE).upsert(
            {"cache_key": _cache_key(namespace, query), "namespace": namespace, "query": query, "response": response}
        ).execute()
    except Exception:
        logger.exception("KV 캐시 저장 실패 - 이미 반환한 응답에는 영향 없음, 무시하고 계속")


async def semantic_get(namespace: str, query: str) -> dict | None:
    """의미가 유사한(문장이 완전히 같지 않아도) 과거 질의의 캐시된 응답.
    임베딩 계산/RPC 실패, 임베딩 API 미설정이면 None."""
    client = await _get_client()
    if client is None:
        return None
    vectors = await embeddings.embed([query])
    if vectors is None:
        return None
    try:
        result = await client.rpc(
            SEMANTIC_MATCH_FN,
            {
                "p_namespace": namespace,
                "query_embedding": vectors[0],
                "match_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
                "match_count": 1,
            },
        ).execute()
        rows = result.data or []
        return rows[0]["response"] if rows else None
    except Exception:
        logger.exception("시맨틱 캐시 조회 실패 - 캐시 미스로 처리")
        return None


async def semantic_set(namespace: str, query: str, response: dict) -> None:
    client = await _get_client()
    if client is None:
        return
    vectors = await embeddings.embed([query])
    if vectors is None:
        return
    try:
        await client.table(SEMANTIC_TABLE).insert(
            {"namespace": namespace, "query": query, "embedding": vectors[0], "response": response}
        ).execute()
    except Exception:
        logger.exception("시맨틱 캐시 저장 실패 - 이미 반환한 응답에는 영향 없음, 무시하고 계속")
