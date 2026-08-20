-- app/llm_cache.py가 쓰는 스키마. Supabase SQL Editor에서 한 번 실행할 것.
-- SUPABASE_URL/SUPABASE_KEY(.env)를 채우기 전까지는 앱이 이 테이블 없이도
-- 그대로 동작한다(캐시가 항상 미스로 처리됨) - 이 마이그레이션은 캐시를
-- 실제로 활성화할 때만 필요하다.

create extension if not exists vector;

-- 1) KV 캐시 - namespace+query가 완전히 같을 때만 재사용.
create table if not exists llm_cache (
  cache_key text primary key,
  namespace text not null,
  query text not null,
  response jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists llm_cache_namespace_idx on llm_cache (namespace);

-- 2) Semantic 캐시 - Qwen text-embedding-v3(1024차원) 코사인 유사도로 재사용.
create table if not exists llm_semantic_cache (
  id bigint generated always as identity primary key,
  namespace text not null,
  query text not null,
  embedding vector(1024) not null,
  response jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists llm_semantic_cache_embedding_idx
  on llm_semantic_cache using ivfflat (embedding vector_cosine_ops);

create or replace function match_llm_semantic_cache(
  p_namespace text,
  query_embedding vector(1024),
  match_threshold float,
  match_count int
)
returns table (id bigint, query text, response jsonb, similarity float)
language sql stable
as $$
  select id, query, response, 1 - (embedding <=> query_embedding) as similarity
  from llm_semantic_cache
  where namespace = p_namespace
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
