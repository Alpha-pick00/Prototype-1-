import asyncio
import json

import app.adk_pipeline as adk_pipeline_module
from app.adk_pipeline import (
    _apply_challenge,
    _broad_web_fallback_search,
    _build_decision,
    _build_pipeline,
    _build_style_guide,
    _comparison_page_listing_fallback,
    _danawa_tables_from_state,
    _elevenst_grounded,
    _finalize_with_danawa,
    _format_price_krw,
    _is_danawa_product_url,
    _judge_eligible_proposals,
    _merge_proposals,
    _on_propose_model_error,
    _on_refine_model_error,
    _pick_and_verify_relaxed,
    _pick_elevenst_candidate,
    _relaxed_fallback_decision,
    _skip_challenge_if_all_structured,
    _skip_judge_if_single_candidate,
    _skip_propose_if_elevenst_grounded,
    _urls_needing_challenge_extract,
    _urls_to_extract,
    _verify_relaxed_verdict,
)
from app.agents.base import CHALLENGE_INSTRUCTIONS, build_challenge_prompt
from app.price_table import build_price_table
from app.schemas import ChallengeResult, ChallengeVerdict, Decision, JudgeVerdict, Proposal, SearchResult
from fetchers import elevenst as elevenst_fetcher
from fetchers.danawa import parse_danawa_html

COUPANG_URL = "https://coupang.com/vp/products/1"
ELEVENST_URL = "https://11st.co.kr/products/2"


def _raw_candidate(product_name: str, price_krw: int, url: str) -> dict:
    return {"product_name": product_name, "price_krw": price_krw, "retailer": "쿠팡", "url": url}


def _merged_candidate(url: str, proposed_by: list[str], product_name: str = "무선 마우스") -> dict:
    return {
        "product_name": product_name,
        "price_krw": 12900,
        "url": url,
        "retailer": "쿠팡",
        "reasons": ["근거"],
        "proposed_by": proposed_by,
        "signals": {},
        "final_score": 0.0,
        "flags": {"shared_url": False, "shared_url_count": 0},
    }


# --- _format_price_krw -------------------------------------------------


def test_format_price_krw_with_value():
    assert _format_price_krw(12900) == "12,900원"


def test_format_price_krw_none():
    assert _format_price_krw(None) == ""


# --- _merge_proposals ----------------------------------------------------


def test_merge_proposals_combines_all_agents():
    raw_by_agent = {
        "gpt": json.dumps([_raw_candidate("무선 마우스", 12900, COUPANG_URL)]),
        "groq": json.dumps([_raw_candidate("무선 마우스", 12900, COUPANG_URL)]),
        "deepseek": json.dumps([]),
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, []))

    assert len(merged) == 1
    assert merged[0]["proposed_by"] == ["gpt", "groq"]


def test_merge_proposals_defaults_to_one_candidate_per_agent():
    """기본값(max_candidates_per_agent 생략)은 지금까지처럼 에이전트당 1개로
    자른다 - 2026-08-15 사용자 요청("최종 후보도 1개만")이 다른 모든
    카테고리의 기본 동작으로 남아있어야 한다(회귀 확인)."""
    raw_by_agent = {
        "gpt": json.dumps(
            [
                _raw_candidate("상품 A", 10000, "https://coupang.com/vp/products/a"),
                _raw_candidate("상품 B", 20000, "https://coupang.com/vp/products/b"),
            ]
        ),
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, []))

    assert len(merged) == 1


def test_merge_proposals_style_guide_mode_keeps_multiple_candidates_per_agent():
    """스타일 가이드 모드(max_candidates_per_agent를 크게 넘김)는 propose가
    이미 배열로 돌려준 여러 후보를 그대로 살려야 한다 - 새 LLM 호출 없이
    후보 풀만 넓어진다."""
    raw_by_agent = {
        "gpt": json.dumps(
            [
                _raw_candidate("상품 A", 10000, "https://coupang.com/vp/products/a"),
                _raw_candidate("상품 B", 20000, "https://coupang.com/vp/products/b"),
                _raw_candidate("상품 C", 30000, "https://coupang.com/vp/products/c"),
            ]
        ),
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, [], max_candidates_per_agent=4))

    assert len(merged) == 3


def test_merge_proposals_skips_agent_with_malformed_json():
    raw_by_agent = {
        "gpt": json.dumps([_raw_candidate("무선 마우스", 12900, COUPANG_URL)]),
        "groq": "이건 JSON이 아닙니다",
        "deepseek": None,
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, []))

    assert len(merged) == 1
    assert merged[0]["proposed_by"] == ["gpt"]


def test_merge_proposals_all_empty_returns_empty_list():
    raw_by_agent = {"gpt": json.dumps([]), "groq": None, "deepseek": ""}

    assert asyncio.run(_merge_proposals(raw_by_agent, [])) == []


def test_merge_proposals_filters_generic_listing_url():
    raw_by_agent = {
        "gpt": json.dumps([_raw_candidate("무선 마우스", 12900, "https://coupang.com/search?q=마우스")]),
        "groq": None,
        "deepseek": None,
    }

    assert asyncio.run(_merge_proposals(raw_by_agent, [])) == []


def test_merge_proposals_filters_danawa_comparison_page_url():
    """2026-08-16, 그라운딩 회귀 파일럿에서 발견: Qwen·DeepSeek이 다나와
    가격비교 페이지 자체(prod.danawa.com/info?pcode=...)를 후보로 제안하면서
    retailer="다나와"(판매처가 아니라 가격비교 사이트 자신), price=""로
    채웠다 - 이 페이지는 실제 구매 가능한 판매처로 연결되지 않으므로
    애초에 후보 풀에 못 들어오게 막는다(진짜 구매 링크인 /bridge/
    loadingBridge.html은 이 패턴에 안 걸려 그대로 통과한다)."""
    raw_by_agent = {
        "gpt": json.dumps([_raw_candidate("위닉스 뽀송 DHC-167IPW", 0, "https://prod.danawa.com/info?pcode=1982936")]),
        "groq": None,
        "deepseek": None,
    }

    assert asyncio.run(_merge_proposals(raw_by_agent, [])) == []


def test_merge_proposals_filters_danawa_mobile_comparison_page_url():
    """2026-08-17, 재검증 파일럿에서 발견: prod.danawa.com/info만 정규식으로
    걸렀더니 같은 문제의 모바일 페이지 변형(m.danawa.com/product/product.html)
    이 그대로 통과했다("LG 그램 16인치 2024" 질의에서 retailer="다나와",
    price="" 재현) - is_danawa_comparison_page를 도메인 기반(다나와 도메인 +
    /bridge/ 아님)으로 일반화한 뒤에는 이 변형도 걸러져야 한다."""
    raw_by_agent = {
        "gpt": json.dumps(
            [_raw_candidate("LG전자 2024 그램16", 0, "https://m.danawa.com/product/product.html?code=45320081")]
        ),
        "groq": None,
        "deepseek": None,
    }

    assert asyncio.run(_merge_proposals(raw_by_agent, [])) == []


def test_merge_proposals_keeps_danawa_bridge_purchase_link():
    """/bridge/loadingBridge.html은 다나와가 최종 판매처로 리다이렉트하는
    실제 구매 링크라 가격비교 페이지 필터에 걸리면 안 된다."""
    bridge_url = "https://prod.danawa.com/bridge/loadingBridge.html?pcode=1&cmpnyc=EE715"
    raw_by_agent = {
        "gpt": json.dumps([_raw_candidate("무선 마우스", 12900, bridge_url)]),
        "groq": None,
        "deepseek": None,
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, []))

    assert len(merged) == 1
    assert merged[0]["url"] == bridge_url


def test_merge_proposals_filters_candidate_with_empty_url():
    """url이 빈 후보를 그대로 통과시키면, 실제로 살 수 있는 페이지가 없는
    후보가 심사까지 흘러가 judge가 존재하지 않는 URL을 스스로 지어내
    채우는 문제로 이어진다 — 애초에 후보 풀에 들어오지 못하게 막는다."""
    raw_by_agent = {
        "gpt": json.dumps([_raw_candidate("무선 마우스", 12900, "")]),
        "groq": None,
        "deepseek": None,
    }

    assert asyncio.run(_merge_proposals(raw_by_agent, [])) == []


# --- _on_propose_model_error ------------------------------------------------


class _StubCallbackContext:
    def __init__(self, agent_name: str = "groq"):
        self.agent_name = agent_name


def test_on_propose_model_error_returns_none_to_propagate_failure():
    """2026-08-18, 사용자 요청: "3개중 하나라도 빠지면 결과를 내지 않도록
    해야지 왜 3개를 안쓰고 2개만해서 결과를 내" - propose 3개(gpt/groq/
    deepseek) 중 하나라도 실패하면 그 실패를 빈 배열로 조용히 덮지 않고
    그대로 흘려보내야 한다. ADK는 이 콜백이 None을 반환하면 원래 예외를
    그대로 raise한다(google.adk.flows.llm_flows.base_llm_flow 참고) -
    run_stream()의 바깥 try/except가 이를 잡아 proposals를 빈 채로 두고
    기존 clarify/relaxed fallback/NO_CANDIDATE_ERROR 경로로 이어지므로,
    2/3만으로 최종 답을 내지 않는다."""
    result = _on_propose_model_error(_StubCallbackContext(), None, RuntimeError("모델 호출 실패"))

    assert result is None


# --- _on_refine_model_error ---------------------------------------------------


def test_on_refine_model_error_returns_none_to_propagate_failure():
    """2026-08-18, 사용자 요청: "AI 모델중에 하나라도 토큰 다쓰면 실행되지
    않도록 바꿔줘" - propose와 같은 원칙을 refine에도 적용한다. 원래는
    정제를 포기하고 원본 질의로 폴백해 계속 진행했는데, 이제는 모델
    하나라도 실패하면 정직하게 전체를 실패시켜야 하므로 None을 반환해
    ADK가 원본 예외를 그대로 raise하게 한다."""
    result = _on_refine_model_error(_StubCallbackContext("refine"), None, RuntimeError("모델 호출 실패"))

    assert result is None


# --- _apply_challenge ------------------------------------------------------


def test_apply_challenge_matches_verdict_by_url_even_when_order_differs():
    candidates = [
        _merged_candidate(COUPANG_URL, ["gpt"], "상품A"),
        _merged_candidate(ELEVENST_URL, ["groq"], "상품B"),
    ]
    # 검증 결과 순서가 후보 순서와 뒤바뀜 — url로 정확히 매칭돼야 한다.
    challenge = ChallengeResult(
        verdicts=[
            ChallengeVerdict(url=ELEVENST_URL, verified=False, note="상품B 우려"),
            ChallengeVerdict(url=COUPANG_URL, verified=True, note="상품A 통과"),
        ]
    )

    proposals = _apply_challenge(candidates, challenge)

    a = next(p for p in proposals if p.url == COUPANG_URL)
    b = next(p for p in proposals if p.url == ELEVENST_URL)
    assert a.verified is True and a.challenge_note == "상품A 통과"
    assert b.verified is False and b.challenge_note == "상품B 우려"


def test_apply_challenge_falls_back_to_index_when_url_missing():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt"])]
    challenge = ChallengeResult(verdicts=[ChallengeVerdict(url=None, verified=True, note="통과")])

    proposals = _apply_challenge(candidates, challenge)

    assert proposals[0].verified is True
    assert proposals[0].challenge_note == "통과"


def test_apply_challenge_empty_verdicts_leaves_all_unverified():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt"]), _merged_candidate(ELEVENST_URL, ["groq"])]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]))

    assert all(p.verified is None and p.challenge_note is None for p in proposals)


def test_apply_challenge_derives_agent_and_proposed_by_from_candidate():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt", "deepseek"])]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]))

    assert proposals[0].agent == "gpt"
    assert proposals[0].proposed_by == ["gpt", "deepseek"]
    assert proposals[0].price == "12,900원"


def test_apply_challenge_empty_candidates_returns_empty_list():
    assert _apply_challenge([], ChallengeResult(verdicts=[])) == []


def test_apply_challenge_uses_refreshed_price_when_verdict_provides_it():
    """2026-08-19, 사용자 리포트("실제로 사이트 들어갔을 때는 다른 가격을
    가져오는 문제") - challenge 직전 _ExtractPagesNode가 이미 라이브로 재조회한
    후보 페이지 원문을 DeepSeek가 읽고 refreshed_price_krw를 채워주면, 그
    갱신된 가격이 새 네트워크 호출 없이 최종 Proposal.price에 반영돼야 한다."""
    candidates = [_merged_candidate(COUPANG_URL, ["gpt"])]
    challenge = ChallengeResult(
        verdicts=[
            ChallengeVerdict(
                url=COUPANG_URL, verified=True, note="상품 일치", refreshed_price_krw=15000
            )
        ]
    )

    proposals = _apply_challenge(candidates, challenge)

    assert proposals[0].price == "15,000원"
    assert "재조회 시점 가격으로 갱신됨" in proposals[0].challenge_note


def test_apply_challenge_keeps_original_price_when_verdict_has_no_refreshed_price():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt"])]
    challenge = ChallengeResult(verdicts=[ChallengeVerdict(url=COUPANG_URL, verified=True, note="통과")])

    proposals = _apply_challenge(candidates, challenge)

    assert proposals[0].price == "12,900원"
    assert proposals[0].challenge_note == "통과"


def test_apply_challenge_ignores_refreshed_price_for_danawa_sourced_candidate():
    """danawa 출신 후보는 원래 가격 자체가 이미 구조화된 실측 스크래핑이라
    DeepSeek 검증(및 refreshed_price_krw)을 아예 안 거친다 - 만약 verdict에
    같은 url로 refreshed_price_krw가 실려와도 무시돼야 한다."""
    candidates = [_merged_candidate(COUPANG_URL, ["danawa"])]
    challenge = ChallengeResult(
        verdicts=[
            ChallengeVerdict(url=COUPANG_URL, verified=False, note="무시돼야 함", refreshed_price_krw=99999)
        ]
    )

    proposals = _apply_challenge(candidates, challenge)

    assert proposals[0].price == "12,900원"
    assert proposals[0].verified is True


def test_apply_challenge_drops_expired_danawa_candidate_entirely():
    """가격비교가 중지된(다나와가 서비스 종료로 표시하는) 페이지는 verified=False로
    남기지 않고 결과에서 아예 빠져야 한다 — "가격미확인" 카드로 노출되면 안 된다."""
    candidates = [
        _merged_candidate(COUPANG_URL, ["gpt"], "상품A"),
        _merged_candidate(ELEVENST_URL, ["groq"], "상품B"),
    ]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]), {ELEVENST_URL})

    assert len(proposals) == 1
    assert proposals[0].url == COUPANG_URL


def test_apply_challenge_all_candidates_expired_returns_empty_list():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt"])]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]), {COUPANG_URL})

    assert proposals == []


# --- _is_danawa_product_url -------------------------------------------------


def test_is_danawa_product_url_matches_prod_danawa():
    assert _is_danawa_product_url("https://prod.danawa.com/info/?pcode=12345") is True


def test_is_danawa_product_url_rejects_other_domains():
    assert _is_danawa_product_url(COUPANG_URL) is False


def test_is_danawa_product_url_rejects_none_and_empty():
    assert _is_danawa_product_url(None) is False
    assert _is_danawa_product_url("") is False


# --- _build_decision (judge의 자유 텍스트보다 그라운딩된 후보 데이터를 우선) ---


def _proposal_for_decision(url: str, price: str = "12,900원", retailer: str = "쿠팡") -> Proposal:
    return Proposal(
        agent="gpt", product_name="무선 마우스", price=price, retailer=retailer, url=url, verified=True
    )


def test_build_decision_prefers_matched_candidate_fields_over_judge_raw_text():
    """judge가 raw_decision에 실제 후보와 다른 price/retailer/url을 지어내도
    (예: url이 빈 후보를 골랐을 때 그럴듯한 URL을 스스로 채우는 경우),
    최종 Decision은 실제로 검증된 matched 후보의 값을 써야 한다."""
    proposals = [_proposal_for_decision(COUPANG_URL, price="12,900원", retailer="쿠팡")]
    state = {
        "raw_decision": {
            "product_name": "무선 마우스",
            "price": "9,900원",  # 후보에 없는 값 — judge가 지어냄
            "retailer": "다나와",  # 후보에 없는 값 — judge가 지어냄
            "url": COUPANG_URL,
            "reasoning": "가장 저렴합니다.",
        }
    }

    decision = _build_decision(state, proposals)

    assert decision.price == "12,900원"
    assert decision.retailer == "쿠팡"
    assert decision.url == COUPANG_URL


def test_build_decision_propagates_verified_from_matched_proposal():
    """Decision.verified(2026-08-16 추가)는 matched proposal의 challenge 검증
    결과를 그대로 물려받아야 한다 - 프론트/API 소비자가 이 답이 실제로
    그라운딩 검증을 통과했는지 알 수 있게 하기 위함."""
    proposals = [_proposal_for_decision(COUPANG_URL)]  # verified=True 고정
    state = {
        "raw_decision": {
            "product_name": "무선 마우스",
            "price": "12,900원",
            "retailer": "쿠팡",
            "url": COUPANG_URL,
            "reasoning": "가장 저렴합니다.",
        }
    }

    decision = _build_decision(state, proposals)

    assert decision.verified is True


def test_build_decision_falls_back_to_raw_when_matched_field_missing():
    proposals = [_proposal_for_decision(COUPANG_URL, price="", retailer="")]
    state = {
        "raw_decision": {
            "product_name": "무선 마우스",
            "price": "12,900원",
            "retailer": "쿠팡",
            "url": COUPANG_URL,
            "reasoning": "가장 저렴합니다.",
        }
    }

    decision = _build_decision(state, proposals)

    assert decision.price == "12,900원"
    assert decision.retailer == "쿠팡"


def test_build_decision_returns_none_without_raw_decision():
    assert _build_decision({}, [_proposal_for_decision(COUPANG_URL)]) is None


def test_build_decision_returns_none_without_proposals():
    assert _build_decision({"raw_decision": {"url": COUPANG_URL}}, []) is None


# --- _build_style_guide (취향 주도 카테고리에서 challenge를 통과한 proposals를
# 스타일별로 그룹핑, 2026-08-19 사용자 요청: GPT 쇼핑처럼 여러 검증된 후보를
# 스타일별로 보여주되 최종 추천은 지금처럼 judge가 고른 하나를 그대로 쓴다) ---


def test_build_style_guide_returns_none_with_fewer_than_two_eligible_proposals():
    proposals = [_proposal_for_decision(COUPANG_URL)]
    assert asyncio.run(_build_style_guide("스니커즈", proposals)) is None


def test_build_style_guide_grounds_groups_to_real_proposal_urls(monkeypatch):
    """LLM이 목록에 없는 url을 지어내면(judge에서 이미 확인된 위험과 동일한
    패턴) 그 그룹은 조용히 버려야 한다 - style_guide도 judge의 _build_decision과
    같은 그라운딩 원칙을 지킨다."""
    proposals = [
        _proposal_for_decision(COUPANG_URL, price="201,700원", retailer="쿠팡"),
        _proposal_for_decision(ELEVENST_URL, price="181,300원", retailer="11번가"),
    ]

    async def _fake_generate(query, eligible):
        return {
            "intro": "비즈니스 캐주얼에는 이런 스니커즈가 좋아요.",
            "groups": [
                {"label": "가장 무난한 선택", "description": "화이트 가죽 미니멀 스니커즈.", "url": COUPANG_URL},
                {"label": "지어낸 후보", "description": "목록에 없는 url.", "url": "https://example.com/fake"},
            ],
            "closing_pick": "화이트 가죽이 1순위예요.",
        }

    monkeypatch.setattr(adk_pipeline_module.judge_module, "generate_style_guide", _fake_generate)

    style_guide = asyncio.run(_build_style_guide("비즈니스 캐주얼 스니커즈", proposals))

    # 그라운딩된 그룹이 하나뿐이라(지어낸 url은 버려짐) 2개 미만 -> None.
    assert style_guide is None


def test_build_style_guide_builds_from_grounded_groups(monkeypatch):
    proposals = [
        _proposal_for_decision(COUPANG_URL, price="201,700원", retailer="쿠팡"),
        _proposal_for_decision(ELEVENST_URL, price="181,300원", retailer="11번가"),
    ]

    async def _fake_generate(query, eligible):
        return {
            "intro": "비즈니스 캐주얼에는 이런 스니커즈가 좋아요.",
            "groups": [
                {"label": "가장 무난한 선택", "description": "화이트 가죽 미니멀 스니커즈.", "url": COUPANG_URL},
                {"label": "좀 더 캐주얼하게", "description": "테니스화 계열.", "url": ELEVENST_URL},
            ],
            "closing_pick": "화이트 가죽이 1순위예요.",
        }

    monkeypatch.setattr(adk_pipeline_module.judge_module, "generate_style_guide", _fake_generate)

    style_guide = asyncio.run(_build_style_guide("비즈니스 캐주얼 스니커즈", proposals))

    assert style_guide is not None
    assert style_guide.intro == "비즈니스 캐주얼에는 이런 스니커즈가 좋아요."
    assert [g.url for g in style_guide.groups] == [COUPANG_URL, ELEVENST_URL]
    assert style_guide.closing_pick == "화이트 가죽이 1순위예요."


def test_build_style_guide_returns_none_when_generation_fails(monkeypatch):
    proposals = [
        _proposal_for_decision(COUPANG_URL),
        _proposal_for_decision(ELEVENST_URL),
    ]

    async def _boom(query, eligible):
        raise RuntimeError("groq down")

    monkeypatch.setattr(adk_pipeline_module.judge_module, "generate_style_guide", _boom)

    assert asyncio.run(_build_style_guide("스니커즈", proposals)) is None


# --- _broad_web_fallback_search (다나와 한정 검색이 빈손일 때의 최후 폴백,
# 2026-08-19 사용자 요청: "검색 알고리즘으로 적절한 상품을 찾을 수 없는
# 경우에는 구글 쇼핑에서 사용자 쿼리를 따로 검색해서 상위 5개의 제품을
# 다나와에서 가져오게") -----------------------------------------------------


def test_broad_web_fallback_search_returns_empty_when_no_broad_results(monkeypatch):
    async def _empty_unrestricted(query):
        return []

    monkeypatch.setattr(adk_pipeline_module.search_module, "search_unrestricted", _empty_unrestricted)

    results = asyncio.run(_broad_web_fallback_search("존재하지 않는 상품명 xyz"))

    assert results == []


def _elevenst_product(name: str, code: str = "1", detail_url: str | None = None) -> elevenst_fetcher.Product:
    return elevenst_fetcher.Product(
        code=code,
        name=name,
        price=10000,
        sale_price=9000,
        image_url=None,
        seller_nick="테스트샵",
        detail_url=detail_url or f"https://www.11st.co.kr/products/{code}",
        delivery=None,
        review_count=None,
        buy_satisfy=None,
        discount=None,
    )


def test_broad_web_fallback_search_regrounds_via_elevenst_using_top_result_title(monkeypatch):
    """비제한 검색(구글 쇼핑 대체)이 찾은 1순위 결과의 제목을 실제 상품명
    삼아 11번가에 딱 한 번만 재검색해, 그 결과를 11번가 URL을 가진
    SearchResult로 바꿔야 한다(2026-08-20, 다나와 배제 이후에도 이 최후
    폴백이 다나와 URL을 다시 만들어내면 안 됨)."""
    captured: dict = {}

    async def _fake_unrestricted(query):
        return [
            SearchResult(title="샤오미 15T 프로 512GB", url="https://example.com/a", snippet="..."),
            SearchResult(title="다른 후보", url="https://example.com/b", snippet="..."),
        ]

    async def _fake_search_products(api_key, keyword, page_size=20, **kwargs):
        captured["api_key"] = api_key
        captured["keyword"] = keyword
        captured["page_size"] = page_size
        return elevenst_fetcher.SearchResult(
            total_count=2,
            products=[
                _elevenst_product("샤오미 15T 프로 512GB 블랙", code="111"),
                _elevenst_product("샤오미 15T 프로 512GB 화이트", code="222"),
            ],
            categories=[],
        )

    monkeypatch.setattr(adk_pipeline_module.settings, "elevenst_api_key", "test-key")
    monkeypatch.setattr(adk_pipeline_module.search_module, "search_unrestricted", _fake_unrestricted)
    monkeypatch.setattr(adk_pipeline_module.elevenst_fetcher, "search_products", _fake_search_products)

    results = asyncio.run(_broad_web_fallback_search("애매한 원래 질의"))

    assert captured["keyword"] == "샤오미 15T 프로 512GB"
    assert captured["page_size"] == adk_pipeline_module._BROAD_FALLBACK_ELEVENST_LIMIT
    assert [r.url for r in results] == [
        "https://www.11st.co.kr/products/111",
        "https://www.11st.co.kr/products/222",
    ]
    assert results[0].title == "샤오미 15T 프로 512GB 블랙"


def test_broad_web_fallback_search_returns_empty_when_elevenst_key_missing(monkeypatch):
    async def _fake_unrestricted(query):
        return [SearchResult(title="샤오미 15T 프로", url="https://example.com/a", snippet="...")]

    monkeypatch.setattr(adk_pipeline_module.settings, "elevenst_api_key", None)
    monkeypatch.setattr(adk_pipeline_module.search_module, "search_unrestricted", _fake_unrestricted)

    results = asyncio.run(_broad_web_fallback_search("애매한 원래 질의"))

    assert results == []


def test_broad_web_fallback_search_returns_empty_when_elevenst_lookup_fails(monkeypatch):
    async def _fake_unrestricted(query):
        return [SearchResult(title="샤오미 15T 프로", url="https://example.com/a", snippet="...")]

    async def _boom(api_key, keyword, page_size=20, **kwargs):
        raise elevenst_fetcher.ElevenstApiError("11번가 검색 차단")

    monkeypatch.setattr(adk_pipeline_module.settings, "elevenst_api_key", "test-key")
    monkeypatch.setattr(adk_pipeline_module.search_module, "search_unrestricted", _fake_unrestricted)
    monkeypatch.setattr(adk_pipeline_module.elevenst_fetcher, "search_products", _boom)

    results = asyncio.run(_broad_web_fallback_search("애매한 원래 질의"))

    assert results == []


# --- _pick_elevenst_candidate (_ElevenstFetchNode의 순수 후보 선정 로직) ----


def test_pick_elevenst_candidate_rejects_ungrounded_products():
    """query와 전혀 관련 없는 상품만 있으면(그라운딩 게이트 실패) 후보를
    만들지 않는다 - _DanawaFetchNode의 2026-08-16 하드닝과 같은 이유."""
    result = elevenst_fetcher.SearchResult(
        total_count=1, products=[_elevenst_product("완전히 다른 상품 아이패드 케이스")], categories=[]
    )

    assert _pick_elevenst_candidate("삼성전자 갤럭시 버즈3 프로", result) is None


def test_pick_elevenst_candidate_picks_cheapest_among_matches():
    cheap = elevenst_fetcher.Product(
        code="1", name="무선 마우스 A", price=15000, sale_price=12000, image_url=None,
        seller_nick="샵A", detail_url="https://www.11st.co.kr/products/1", delivery=None,
        review_count=None, buy_satisfy=None, discount=None,
    )
    expensive = elevenst_fetcher.Product(
        code="2", name="무선 마우스 B", price=20000, sale_price=18000, image_url=None,
        seller_nick="샵B", detail_url="https://www.11st.co.kr/products/2", delivery=None,
        review_count=None, buy_satisfy=None, discount=None,
    )
    result = elevenst_fetcher.SearchResult(total_count=2, products=[expensive, cheap], categories=[])

    candidate = _pick_elevenst_candidate("무선 마우스", result)

    assert candidate is not None
    assert candidate.price_krw == 12000
    assert candidate.url == "https://www.11st.co.kr/products/1"
    assert candidate.retailer == "11번가"


def test_pick_elevenst_candidate_excludes_products_without_detail_url():
    no_url = elevenst_fetcher.Product(
        code="1", name="무선 마우스", price=5000, sale_price=None, image_url=None,
        seller_nick="샵A", detail_url=None, delivery=None, review_count=None,
        buy_satisfy=None, discount=None,
    )
    result = elevenst_fetcher.SearchResult(total_count=1, products=[no_url], categories=[])

    assert _pick_elevenst_candidate("무선 마우스", result) is None


def test_pick_elevenst_candidate_excludes_products_without_price():
    no_price = elevenst_fetcher.Product(
        code="1", name="무선 마우스", price=None, sale_price=None, image_url=None,
        seller_nick="샵A", detail_url="https://www.11st.co.kr/products/1", delivery=None,
        review_count=None, buy_satisfy=None, discount=None,
    )
    result = elevenst_fetcher.SearchResult(total_count=1, products=[no_price], categories=[])

    assert _pick_elevenst_candidate("무선 마우스", result) is None


def test_pick_elevenst_candidate_empty_products_returns_none():
    result = elevenst_fetcher.SearchResult(total_count=0, products=[], categories=[])

    assert _pick_elevenst_candidate("무선 마우스", result) is None


# --- _urls_to_extract (challenge 전 실제 페이지 재조회 대상) ----------------


def test_urls_to_extract_returns_candidate_urls():
    candidates = [
        {"url": COUPANG_URL},
        {"url": ELEVENST_URL},
    ]
    assert _urls_to_extract(candidates) == [COUPANG_URL, ELEVENST_URL]


def test_urls_to_extract_skips_candidates_without_url():
    candidates = [{"url": None}, {"url": COUPANG_URL}, {}]
    assert _urls_to_extract(candidates) == [COUPANG_URL]


def test_urls_to_extract_caps_at_max_candidates():
    candidates = [{"url": f"https://coupang.com/vp/products/{i}"} for i in range(20)]
    urls = _urls_to_extract(candidates)
    assert len(urls) == 10


# --- _urls_needing_challenge_extract (danawa 픽은 challenge 검증을 안 쓰므로 제외) --


def test_urls_needing_challenge_extract_excludes_danawa_proposed_candidate():
    candidates = [
        {"url": COUPANG_URL, "proposed_by": ["gpt"]},
        {"url": ELEVENST_URL, "proposed_by": ["danawa"]},
    ]
    assert _urls_needing_challenge_extract(candidates) == [COUPANG_URL]


def test_urls_needing_challenge_extract_keeps_danawa_domain_url_not_proposed_by_danawa_node():
    danawa_url = "https://prod.danawa.com/info/?pcode=1"
    candidates = [{"url": danawa_url, "proposed_by": ["gpt", "deepseek"]}]
    assert _urls_needing_challenge_extract(candidates) == [danawa_url]


def test_urls_needing_challenge_extract_excludes_when_danawa_merged_with_other_agents():
    candidates = [{"url": COUPANG_URL, "proposed_by": ["gpt", "danawa"]}]
    assert _urls_needing_challenge_extract(candidates) == []


def test_urls_needing_challenge_extract_handles_missing_proposed_by():
    candidates = [{"url": COUPANG_URL}]
    assert _urls_needing_challenge_extract(candidates) == [COUPANG_URL]


def test_urls_needing_challenge_extract_excludes_elevenst_proposed_candidate():
    """2026-08-20 - elevenst 픽도 다나와와 같은 이유(구조화 소스라 challenge
    검증 결과가 버려짐)로 extract() 대상에서 빠져야 한다."""
    candidates = [
        {"url": COUPANG_URL, "proposed_by": ["gpt"]},
        {"url": ELEVENST_URL, "proposed_by": ["elevenst"]},
    ]
    assert _urls_needing_challenge_extract(candidates) == [COUPANG_URL]


# --- _judge_eligible_proposals (verified=False 후보 judge 이전 필터링) -----


def _proposal(url: str, verified: bool | None) -> Proposal:
    return Proposal(agent="gpt", product_name="상품", price="1,000원", retailer="쿠팡", url=url, verified=verified)


def test_judge_eligible_proposals_filters_out_verified_false():
    proposals = [_proposal(COUPANG_URL, True), _proposal(ELEVENST_URL, False)]

    eligible = _judge_eligible_proposals(proposals)

    assert [p.url for p in eligible] == [COUPANG_URL]


def test_judge_eligible_proposals_keeps_unverified_candidates():
    proposals = [_proposal(COUPANG_URL, None)]

    assert _judge_eligible_proposals(proposals) == proposals


def test_judge_eligible_proposals_falls_back_to_full_list_when_all_rejected():
    proposals = [_proposal(COUPANG_URL, False), _proposal(ELEVENST_URL, False)]

    assert _judge_eligible_proposals(proposals) == proposals


# --- 다나와 실측가 주입(_DanawaFetchNode 포팅) ------------------------------
# PRESERVED FROM seungmin/lsm의 run_single_debate_price_table_variant를
# ADK 파이프라인으로 포팅(2026-08-16) - tests/test_pipeline_danawa.py의
# _danawa_html/_offer_li와 같은 합성 HTML 픽스처 패턴을 그대로 쓴다.


def test_merge_proposals_includes_danawa_agent():
    raw_by_agent = {
        "gpt": None,
        "groq": None,
        "deepseek": None,
        "danawa": json.dumps([_raw_candidate("무선 마우스", 12900, COUPANG_URL)]),
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, []))

    assert len(merged) == 1
    assert merged[0]["proposed_by"] == ["danawa"]


def test_merge_proposals_includes_elevenst_agent():
    """2026-08-20 - _FilterMergeNode의 raw_by_agent에 elevenst_raw가 흘러들어가면
    다른 3개 슬롯과 동일하게 병합 풀에 합류해야 한다."""
    raw_by_agent = {
        "gpt": None,
        "groq": None,
        "deepseek": None,
        "elevenst": json.dumps([_raw_candidate("무선 마우스", 12900, ELEVENST_URL)]),
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, []))

    assert len(merged) == 1
    assert merged[0]["proposed_by"] == ["elevenst"]


def test_merge_proposals_resolves_comparison_page_to_bridge_url_when_pcode_matches():
    """2026-08-18, 그라운딩 회귀 파일럿에서 발견: Qwen/Groq/DeepSeek이 다나와
    단독 검색 결과에서 고를 수 있는 URL은 거의 전부 가격비교 페이지
    (prod.danawa.com/info?pcode=...) 형태뿐인데, 해석 없이 필터링만 하면 세
    제안자가 후보를 거의 못 만든다. 같은 propose 라운드에서 _DanawaFetchNode가
    이미 페치해 둔 가격표(danawa_tables)에 같은 pcode가 있으면, 그 A등급
    최저가 구매링크로 바꿔치기해 후보를 살려야 한다."""
    table, result = _danawa_price_table_pair(
        "레이저 데스에더 V3", [_offer_li("옥션", "45,000", "EE715", link_pcode="777")], pcode="19505813"
    )
    raw_by_agent = {
        "gpt": json.dumps(
            [_raw_candidate("Razer DeathAdder V3 (정품)", 0, "https://prod.danawa.com/info?pcode=19505813")]
        ),
        "groq": None,
        "deepseek": None,
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, [(table, result)]))

    assert len(merged) == 1
    assert merged[0]["url"] == "https://prod.danawa.com/bridge/loadingBridge.html?cmpnyc=EE715&link_pcode=777"
    assert merged[0]["price_krw"] == 45000
    assert merged[0]["retailer"] == "옥션"


def test_merge_proposals_still_filters_comparison_page_when_pcode_does_not_match():
    """danawa_tables에 같은 pcode를 가진 가격표가 없으면(예: _DanawaFetchNode가
    다른 상품을 찾았거나 아예 실패했으면) 해석할 근거가 없다 - 안 검증된
    값을 지어내지 않고 기존처럼 그대로 걸러야 한다."""
    table, result = _danawa_price_table_pair(
        "다른 상품", [_offer_li("옥션", "45,000", "EE715", link_pcode="777")], pcode="99999999"
    )
    raw_by_agent = {
        "gpt": json.dumps(
            [_raw_candidate("Razer DeathAdder V3 (정품)", 0, "https://prod.danawa.com/info?pcode=19505813")]
        ),
        "groq": None,
        "deepseek": None,
    }

    merged = asyncio.run(_merge_proposals(raw_by_agent, [(table, result)]))

    assert merged == []


def test_apply_challenge_marks_danawa_sourced_candidate_verified_without_challenge_verdict():
    """다나와 실측가는 이미 검증된 데이터라, DeepSeek 검증 결과가 하나도
    없어도(verdicts=[]) verified=None(미검증)이 아니라 True로 강제돼야 한다."""
    candidates = [_merged_candidate(COUPANG_URL, ["danawa"], "상품A")]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]))

    assert proposals[0].verified is True
    assert "다나와" in proposals[0].challenge_note


def test_apply_challenge_danawa_consensus_candidate_ignores_deepseek_verdict():
    """다른 에이전트와 합의(병합)돼도 proposed_by에 danawa가 있으면 verified=True다 -
    DeepSeek이 그 URL을 우려로 표시했더라도 실측가가 있으면 덮어쓴다."""
    candidates = [_merged_candidate(COUPANG_URL, ["gpt", "danawa"], "상품A")]
    challenge = ChallengeResult(verdicts=[ChallengeVerdict(url=COUPANG_URL, verified=False, note="우려")])

    proposals = _apply_challenge(candidates, challenge)

    assert proposals[0].verified is True
    assert proposals[0].challenge_note != "우려"


def test_apply_challenge_non_danawa_candidate_unaffected_by_danawa_override():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt"], "상품A")]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]))

    assert proposals[0].verified is None
    assert proposals[0].challenge_note is None


def test_apply_challenge_marks_elevenst_sourced_candidate_verified_without_challenge_verdict():
    """2026-08-20 - 11번가 공식 API 구조화 데이터도 다나와와 같은 이유로
    challenge 검증 없이 verified=True로 강제돼야 한다."""
    candidates = [_merged_candidate(ELEVENST_URL, ["elevenst"], "상품A")]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]))

    assert proposals[0].verified is True
    assert "11번가" in proposals[0].challenge_note


def test_apply_challenge_danawa_and_elevenst_merged_candidate_verified():
    """다나와 픽과 11번가 픽이 같은 상품으로 병합되면(fusion.dedup이 상품명
    유사도만으로도 병합할 수 있음) 두 소스 이름이 모두 challenge_note에
    드러나야 한다."""
    candidates = [_merged_candidate(COUPANG_URL, ["danawa", "elevenst"], "상품A")]

    proposals = _apply_challenge(candidates, ChallengeResult(verdicts=[]))

    assert proposals[0].verified is True
    assert "다나와" in proposals[0].challenge_note
    assert "11번가" in proposals[0].challenge_note


# --- 쿠팡 교차 확인(build_challenge_prompt, 2026-08-16) ----------------------


def test_build_challenge_prompt_without_coupang_results_matches_prior_output():
    without_coupang = build_challenge_prompt("무선 마우스", [], [])
    with_empty_list = build_challenge_prompt("무선 마우스", [], [], None, [])

    assert without_coupang == with_empty_list
    # 결과 블록 자체는 안 붙지만, CHALLENGE_INSTRUCTIONS의 사용법 설명 문구는 항상 포함된다.
    assert "쿠팡 교차 확인 검색 결과(참고용)" not in without_coupang


def test_build_challenge_prompt_includes_coupang_block_when_provided():
    coupang_results = [SearchResult(title="쿠팡 무선 마우스", url=COUPANG_URL, snippet="12,900원")]

    prompt = build_challenge_prompt("무선 마우스", [], [], None, coupang_results)

    assert "쿠팡 교차 확인 검색 결과(참고용)" in prompt
    assert COUPANG_URL in prompt


def test_challenge_instructions_treat_coupang_signal_as_soft():
    assert "곧바로 false로 판단하지 마세요" in CHALLENGE_INSTRUCTIONS
    assert "쿠팡" in CHALLENGE_INSTRUCTIONS


# --- 네이버쇼핑 교차 확인(build_challenge_prompt, 2026-08-16) ----------------


def test_build_challenge_prompt_without_naver_results_matches_prior_output():
    without_naver = build_challenge_prompt("무선 마우스", [], [])
    with_empty_list = build_challenge_prompt("무선 마우스", [], [], None, None, [])

    assert without_naver == with_empty_list
    assert "네이버쇼핑 교차 확인 검색 결과(참고용)" not in without_naver


def test_build_challenge_prompt_includes_naver_block_when_provided():
    naver_url = "https://shopping.naver.com/products/1"
    naver_results = [SearchResult(title="네이버 무선 마우스", url=naver_url, snippet="12,900원")]

    prompt = build_challenge_prompt("무선 마우스", [], [], None, None, naver_results)

    assert "네이버쇼핑 교차 확인 검색 결과(참고용)" in prompt
    assert naver_url in prompt


def test_build_challenge_prompt_includes_both_coupang_and_naver_blocks():
    coupang_results = [SearchResult(title="쿠팡", url=COUPANG_URL, snippet="12,900원")]
    naver_results = [SearchResult(title="네이버", url="https://shopping.naver.com/products/1", snippet="12,900원")]

    prompt = build_challenge_prompt("무선 마우스", [], [], None, coupang_results, naver_results)

    assert "쿠팡 교차 확인 검색 결과(참고용)" in prompt
    assert "네이버쇼핑 교차 확인 검색 결과(참고용)" in prompt


def test_challenge_instructions_treat_naver_signal_as_soft():
    assert "곧바로 false로 판단하지 마세요" in CHALLENGE_INSTRUCTIONS
    assert "네이버쇼핑" in CHALLENGE_INSTRUCTIONS


def _offer_li(alt: str, price_text: str, cmpnyc: str, link_pcode: str = "999") -> str:
    return f"""
    <li class="list-item">
      <div class="box__logo"><img src="x.png" alt="{alt}"></div>
      <div class="box__price"><div class="sell-price"><span class="text__num">{price_text}</span></div></div>
      <div class="box__delivery">무료배송</div>
      <a class="link__full-cover" href="https://prod.danawa.com/bridge/loadingBridge.html?cmpnyc={cmpnyc}&link_pcode={link_pcode}"></a>
    </li>
    """


def _danawa_html(product_name: str, offers_html: list[str]) -> str:
    offers_block = "".join(offers_html)
    return f"""
    <html><body>
    <img alt="{product_name}_이미지" src="x.png">
    <ul class="list__mall-price">{offers_block}</ul>
    </body></html>
    """


def _danawa_price_table_pair(product_name: str, offers_html: list[str], pcode: str = "1"):
    html = _danawa_html(product_name, offers_html)
    result = parse_danawa_html(f"https://prod.danawa.com/info?pcode={pcode}", html)
    return build_price_table(result), result


def test_danawa_tables_from_state_round_trips_price_table():
    table, result = _danawa_price_table_pair(
        "테스트 상품", [_offer_li("쿠팡", "23,000", "TP40F", link_pcode="1")]
    )
    state = {"danawa_tables": [[table.model_dump(), result]]}

    restored = _danawa_tables_from_state(state)

    assert len(restored) == 1
    restored_table, restored_result = restored[0]
    assert restored_table.product_name == "테스트 상품"
    assert restored_result["product_name"] == "테스트 상품"


# --- _comparison_page_listing_fallback (구매 링크 후보가 전혀 없을 때의 최후
# 폴백, 2026-08-19 사용자 요청: "그래도 추천해줘라고 했을때 답변을 잘해주는거잖아") ---


def test_comparison_page_listing_fallback_returns_none_without_tables():
    assert asyncio.run(_comparison_page_listing_fallback("이어폰", [])) is None


def test_comparison_page_listing_fallback_skips_irrelevant_table_and_uses_first_relevant_one(monkeypatch):
    """CMPNYC_MAP에 구매 링크 규칙이 없는 판매처(쿠팡 - url_rule=None, 실측
    2026-08-19: 다나와-쿠팡 제휴 코드가 막혀 있음)만 걸려도, 실측 가격표
    자체는 있으니 "아무것도 못 찾았다"가 아니라 다나와 비교 페이지를
    정직하게 최종 답으로 낸다. 순서나 가격으로 표를 고르지 않는다 - 회귀
    테스트(2026-08-19 실측: 최저가/첫 순위 휴리스틱 둘 다 "10만원대
    이어폰" 검색에서 무관한 상품(쌍안경, 노트북)을 골랐다) - 각 표마다
    실제로 관련성을 확인해, 무관한 표는 건너뛰고 관련 있는 첫 표만 쓴다."""
    irrelevant_table, irrelevant_result = _danawa_price_table_pair(
        "니쿠라 10-30x25 (쌍안경)", [_offer_li("쿠팡", "5,000", "TP40F")], pcode="111"
    )
    relevant_table, relevant_result = _danawa_price_table_pair(
        "아이리버 무선 이어폰", [_offer_li("쿠팡", "99,000", "TP40F")], pcode="222"
    )

    async def _fake_relevance(query, product_name):
        return product_name == "아이리버 무선 이어폰"

    monkeypatch.setattr(adk_pipeline_module, "_is_relevant_to_query", _fake_relevance)

    decision = asyncio.run(
        _comparison_page_listing_fallback(
            "10만원대 이어폰 추천해줘", [(irrelevant_table, irrelevant_result), (relevant_table, relevant_result)]
        )
    )

    assert decision is not None
    assert decision.product_name == "아이리버 무선 이어폰"
    assert decision.price == "99,000원"
    assert decision.url == "https://prod.danawa.com/info/?pcode=222"
    assert decision.retailer == "다나와 가격비교"
    assert decision.chosen_agent == "danawa"
    assert decision.price_source == "danawa_offer"
    assert decision.verified is True
    assert "다나와가 실측한 가격비교 데이터를" in decision.reasoning


def test_comparison_page_listing_fallback_returns_none_when_all_tables_irrelevant(monkeypatch):
    """전부 무관하면(관련성 판정이 다 false거나 실패하면) 무관한 상품을
    추천하느니 정직하게 포기한다(None -> 호출부가 NO_CANDIDATE_ERROR로
    이어감)."""
    table, result = _danawa_price_table_pair(
        "니쿠라 10-30x25 (쌍안경)", [_offer_li("쿠팡", "5,000", "TP40F")], pcode="111"
    )

    async def _always_irrelevant(query, product_name):
        return False

    monkeypatch.setattr(adk_pipeline_module, "_is_relevant_to_query", _always_irrelevant)

    decision = asyncio.run(_comparison_page_listing_fallback("10만원대 이어폰 추천해줘", [(table, result)]))

    assert decision is None


def test_danawa_tables_from_state_empty_when_missing():
    assert _danawa_tables_from_state({}) == []


def test_finalize_with_danawa_sets_price_source_when_judge_chose_danawa():
    table, result = _danawa_price_table_pair(
        "테스트 상품", [_offer_li("쿠팡", "23,000", "TP40F", link_pcode="1")]
    )
    decision = Decision(
        product_name="테스트 상품",
        price="23,000원",
        retailer="쿠팡",
        url="https://prod.danawa.com/bridge/loadingBridge.html?cmpnyc=TP40F&link_pcode=1",
        reasoning="테스트",
        chosen_agent="danawa",
    )

    updated, price_table = asyncio.run(_finalize_with_danawa(decision, [], [(table, result)]))

    assert updated.price_source == "danawa_offer"
    assert price_table is not None
    assert price_table.product_name == "테스트 상품"


def test_finalize_with_danawa_sets_price_source_when_judge_chose_elevenst():
    """2026-08-20 - judge가 11번가 후보를 골랐고(chosen_agent="elevenst") 매칭되는
    다나와 실측 테이블이 없으면(danawa_tables=[]) price_source가 "llm_guess"가
    아니라 "elevenst_offer"로 남아야 한다."""
    decision = Decision(
        product_name="테스트 상품",
        price="19,000원",
        retailer="11번가",
        url="https://www.11st.co.kr/products/1",
        reasoning="테스트",
        chosen_agent="elevenst",
    )

    updated, price_table = asyncio.run(_finalize_with_danawa(decision, [], []))

    assert updated.price_source == "elevenst_offer"
    assert updated.url == "https://www.11st.co.kr/products/1"
    assert price_table is None


def test_finalize_with_danawa_enriches_matching_llm_decision():
    """judge가 이름이 일치하는 다나와 실측가를 고르지 않았어도(chosen_agent="gpt"),
    상품명이 맞으면 enrich_decision이 가격/URL을 실측치로 덮어쓴다."""
    table, result = _danawa_price_table_pair(
        "테스트 상품", [_offer_li("옥션", "23,000", "EE715", link_pcode="777")]
    )
    decision = Decision(
        product_name="테스트 상품",
        price="가격 정보 없음",
        retailer="다나와",
        url="https://example.com/guess",
        reasoning="테스트",
        chosen_agent="gpt",
    )

    updated, price_table = asyncio.run(_finalize_with_danawa(decision, [], [(table, result)]))

    assert updated.price_source == "danawa_offer"
    assert updated.price == "23,000원"
    assert updated.url == "https://prod.danawa.com/bridge/loadingBridge.html?cmpnyc=EE715&link_pcode=777"
    assert price_table is not None


def test_finalize_with_danawa_leaves_decision_unchanged_when_no_tables():
    decision = Decision(
        product_name="테스트 상품",
        price="10,000원",
        retailer="쿠팡",
        url="https://coupang.com/vp/products/1",
        reasoning="테스트",
        chosen_agent="gpt",
    )

    updated, price_table = asyncio.run(_finalize_with_danawa(decision, [], []))

    assert updated.price_source == "llm_guess"
    assert updated.url == "https://coupang.com/vp/products/1"
    assert price_table is None


# --- _skip_judge_if_single_candidate (judge LLM 호출 생략, 속도 개선) -----------


class _FakeCallbackContext:
    def __init__(self, state: dict):
        self.state = state


def test_skip_judge_returns_verdict_matching_the_only_candidate():
    proposals = [_proposal(COUPANG_URL, True)]
    ctx = _FakeCallbackContext({"proposals": [p.model_dump() for p in proposals]})

    response = _skip_judge_if_single_candidate(ctx, None)

    assert response is not None
    verdict = json.loads(response.content.parts[0].text)
    assert verdict["url"] == COUPANG_URL
    assert verdict["product_name"] == "상품"
    assert verdict["price"] == "1,000원"
    assert verdict["retailer"] == "쿠팡"
    assert verdict["reasoning"]


def test_skip_judge_none_when_no_candidates():
    ctx = _FakeCallbackContext({"proposals": []})

    assert _skip_judge_if_single_candidate(ctx, None) is None


def test_skip_judge_none_when_multiple_candidates():
    proposals = [_proposal(COUPANG_URL, True), _proposal(ELEVENST_URL, True)]
    ctx = _FakeCallbackContext({"proposals": [p.model_dump() for p in proposals]})

    assert _skip_judge_if_single_candidate(ctx, None) is None


def test_skip_judge_uses_only_eligible_candidate_when_others_rejected():
    proposals = [_proposal(COUPANG_URL, True), _proposal(ELEVENST_URL, False)]
    ctx = _FakeCallbackContext({"proposals": [p.model_dump() for p in proposals]})

    response = _skip_judge_if_single_candidate(ctx, None)

    assert response is not None
    verdict = json.loads(response.content.parts[0].text)
    assert verdict["url"] == COUPANG_URL


def test_skip_judge_none_when_the_only_candidate_is_missing_a_required_field():
    incomplete = Proposal(agent="gpt", product_name="상품", price="", retailer="쿠팡", url=COUPANG_URL)
    ctx = _FakeCallbackContext({"proposals": [incomplete.model_dump()]})

    assert _skip_judge_if_single_candidate(ctx, None) is None


# --- _skip_challenge_if_all_structured (challenge LLM 호출 생략, 비용 절감) -----
# (2026-08-20, "LLM이 불필요하게 쓰이고 있는곳" 점검 - _apply_challenge가 구조화
# 소스(danawa/elevenst) 후보의 challenge verdict를 애초에 안 본다는 사실에서 착안)


def test_skip_challenge_returns_empty_verdicts_when_every_candidate_is_structured():
    candidates = [_merged_candidate(ELEVENST_URL, ["elevenst"]), _merged_candidate(COUPANG_URL, ["elevenst", "gpt"])]
    ctx = _FakeCallbackContext({"candidates": candidates})

    response = _skip_challenge_if_all_structured(ctx, None)

    assert response is not None
    assert json.loads(response.content.parts[0].text) == []


def test_skip_challenge_returns_empty_verdicts_when_no_candidates():
    ctx = _FakeCallbackContext({"candidates": []})

    response = _skip_challenge_if_all_structured(ctx, None)

    assert response is not None
    assert json.loads(response.content.parts[0].text) == []


def test_skip_challenge_none_when_a_candidate_is_not_structured():
    """gpt/groq/deepseek만 제안하고 elevenst와 병합되지 않은 후보가 섞여
    있으면(=진짜 검증이 필요한 후보가 있으면) challenge를 그대로 태워야 한다."""
    candidates = [_merged_candidate(ELEVENST_URL, ["elevenst"]), _merged_candidate(COUPANG_URL, ["gpt"])]
    ctx = _FakeCallbackContext({"candidates": candidates})

    assert _skip_challenge_if_all_structured(ctx, None) is None


def test_skip_challenge_none_when_only_candidate_is_llm_proposed():
    candidates = [_merged_candidate(COUPANG_URL, ["gpt", "groq"])]
    ctx = _FakeCallbackContext({"candidates": candidates})

    assert _skip_challenge_if_all_structured(ctx, None) is None


# --- _elevenst_grounded (propose LLM/소프트 교차확인 게이팅의 공유 판정) -------
# (2026-08-20, "3개 LLM까지 필요없다" - elevenst가 propose_parallel보다 먼저
# 순차로 도는 별도 단계가 됐으므로, 그 결과가 state에 최종 반영된 뒤에만
# 안전하게 읽을 수 있다는 전제 하에 이 판정을 공유한다)


def test_elevenst_grounded_true_when_candidate_present():
    assert _elevenst_grounded({"elevenst_raw": json.dumps([_raw_candidate("상품", 1000, ELEVENST_URL)])})


def test_elevenst_grounded_false_when_empty_array():
    assert _elevenst_grounded({"elevenst_raw": "[]"}) is False


def test_elevenst_grounded_false_when_key_missing():
    assert _elevenst_grounded({}) is False


def test_elevenst_grounded_false_when_value_is_malformed_json():
    assert _elevenst_grounded({"elevenst_raw": "not json"}) is False


# --- _skip_propose_if_elevenst_grounded (propose 유일한 LLM을 조건부로만 호출) --


def test_skip_propose_returns_empty_when_elevenst_already_grounded():
    ctx = _FakeCallbackContext({"elevenst_raw": json.dumps([_raw_candidate("상품", 1000, ELEVENST_URL)])})

    response = _skip_propose_if_elevenst_grounded(ctx, None)

    assert response is not None
    assert json.loads(response.content.parts[0].text) == []


def test_skip_propose_none_when_elevenst_grounding_failed():
    """elevenst_raw가 빈 배열(그라운딩 실패)이면 deepseek를 실제로 호출해
    의미적 매칭 안전망을 태워야 한다."""
    ctx = _FakeCallbackContext({"elevenst_raw": "[]"})

    assert _skip_propose_if_elevenst_grounded(ctx, None) is None


def test_skip_propose_none_when_elevenst_raw_missing():
    ctx = _FakeCallbackContext({})

    assert _skip_propose_if_elevenst_grounded(ctx, None) is None


# --- _build_pipeline 구조 (2026-08-20 재구성 회귀 방지) -------------------------


def test_build_pipeline_runs_elevenst_before_propose_and_propose_has_only_deepseek():
    """gpt/groq propose 슬롯 제거 + elevenst를 propose_parallel보다 앞선 순차
    단계로 옮긴 재구성이 실제로 배선됐는지 - _skip_propose_if_elevenst_grounded가
    elevenst_raw를 신뢰성 있게 읽으려면 이 순서가 반드시 지켜져야 한다."""
    pipeline = _build_pipeline()

    top_level = [a.name for a in pipeline.sub_agents]
    assert top_level.index("elevenst") < top_level.index("propose")

    propose = next(a for a in pipeline.sub_agents if a.name == "propose")
    assert [a.name for a in propose.sub_agents] == ["deepseek", "coupang_check", "naver_check"]


# --- relaxed fallback 하드닝(2026-08-16, "구매링크를 안띄워주는거야" 버그의 근본 -----
# 원인이었던 경로 - challenge 검증을 우회할 수 없도록 게이팅한다) ------------------


def _relaxed_verdict(url: str = COUPANG_URL, product_name: str = "무선 마우스") -> JudgeVerdict:
    return JudgeVerdict(
        product_name=product_name, price="12,900원", retailer="쿠팡", url=url, reasoning="가장 관련성이 높습니다."
    )


def _patch_no_cross_check_signals(monkeypatch):
    """쿠팡/네이버 소프트 신호 자체는 이 테스트들의 관심사가 아니므로 항상
    빈 리스트로 고정해 challenge_candidates에 전달되는 인자만 신경 쓴다."""

    async def _empty(query: str) -> list[SearchResult]:
        return []

    monkeypatch.setattr(adk_pipeline_module.search_module, "search_coupang", _empty)
    monkeypatch.setattr(adk_pipeline_module.search_module, "search_naver", _empty)


def test_verify_relaxed_verdict_matches_challenge_verdict_by_url(monkeypatch):
    _patch_no_cross_check_signals(monkeypatch)
    verdict = _relaxed_verdict()

    async def _fake_challenge(query, candidates, search_results, candidate_pages, coupang_results, naver_results):
        return [ChallengeVerdict(url=verdict.url, verified=True, note="검색 결과와 일치")]

    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    verified = asyncio.run(_verify_relaxed_verdict("무선 마우스", verdict, []))

    assert verified is True


def test_verify_relaxed_verdict_returns_none_when_challenge_infra_fails(monkeypatch):
    """challenge_candidates가 빈 리스트(API 오류/파싱 실패)를 돌려주면
    "검증 안 됨"으로 취급해야지, 검증 실패를 "그라운딩 우려"와 혼동해 후보를
    폐기해서는 안 된다."""
    _patch_no_cross_check_signals(monkeypatch)

    async def _fake_challenge(*args, **kwargs):
        return []

    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    verified = asyncio.run(_verify_relaxed_verdict("무선 마우스", _relaxed_verdict(), []))

    assert verified is None


def test_pick_and_verify_relaxed_discards_candidate_rejected_by_challenge(monkeypatch):
    _patch_no_cross_check_signals(monkeypatch)
    verdict = _relaxed_verdict()

    async def _fake_pick(query, search_results):
        return verdict

    async def _fake_challenge(query, candidates, search_results, candidate_pages, coupang_results, naver_results):
        return [ChallengeVerdict(url=verdict.url, verified=False, note="검색 결과 어디에도 이 가격이 없음")]

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)
    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    result = asyncio.run(_pick_and_verify_relaxed("무선 마우스", []))

    assert result is None


def test_pick_and_verify_relaxed_keeps_candidate_when_verified_true(monkeypatch):
    _patch_no_cross_check_signals(monkeypatch)
    verdict = _relaxed_verdict()

    async def _fake_pick(query, search_results):
        return verdict

    async def _fake_challenge(query, candidates, search_results, candidate_pages, coupang_results, naver_results):
        return [ChallengeVerdict(url=verdict.url, verified=True, note="검색 결과와 일치")]

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)
    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    result = asyncio.run(_pick_and_verify_relaxed("무선 마우스", []))

    assert result == (verdict, True)


def test_relaxed_fallback_decision_returns_verified_decision_without_caveat(monkeypatch):
    _patch_no_cross_check_signals(monkeypatch)
    verdict = _relaxed_verdict()

    async def _fake_pick(query, search_results):
        return verdict

    async def _fake_challenge(query, candidates, search_results, candidate_pages, coupang_results, naver_results):
        return [ChallengeVerdict(url=verdict.url, verified=True, note="검색 결과와 일치")]

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)
    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    decision = asyncio.run(_relaxed_fallback_decision("무선 마우스", []))

    assert decision is not None
    assert decision.verified is True
    assert "낮은 확신" not in decision.reasoning


def test_relaxed_fallback_decision_none_when_short_query_rejected_and_cannot_broaden(monkeypatch):
    """질의가 2단어 이하면 broadened_query == query라 재검색을 건너뛴다 - 1라운드가
    challenge에서 명백히 탈락하면 더 시도할 게 없으므로 정직하게 포기해야 한다
    (하드닝 전에는 이 경로가 검증 없이 그대로 최종 응답이 됐다)."""
    _patch_no_cross_check_signals(monkeypatch)
    verdict = _relaxed_verdict()

    async def _fake_pick(query, search_results):
        return verdict

    async def _fake_challenge(query, candidates, search_results, candidate_pages, coupang_results, naver_results):
        return [ChallengeVerdict(url=verdict.url, verified=False, note="검색 결과 어디에도 이 가격이 없음")]

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)
    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    decision = asyncio.run(_relaxed_fallback_decision("마우스", []))

    assert decision is None


def test_relaxed_fallback_decision_broadens_query_after_challenge_rejection(monkeypatch):
    """1라운드 후보가 challenge에서 명백히 탈락하면(verified=False), 완전히
    포기하기 전에 넓힌 질의로 한 번 더 시도한다 - 후보를 아예 못 찾았을 때와
    같은 재시도 경로를 탄다."""
    _patch_no_cross_check_signals(monkeypatch)
    rejected = _relaxed_verdict(url=COUPANG_URL, product_name="무선 마우스 A")
    accepted = _relaxed_verdict(url=ELEVENST_URL, product_name="무선 마우스 B")

    call_count = {"pick": 0}

    async def _fake_pick(query, search_results):
        call_count["pick"] += 1
        return rejected if call_count["pick"] == 1 else accepted

    async def _fake_challenge(query, candidates, search_results, candidate_pages, coupang_results, naver_results):
        url = candidates[0]["url"]
        verified = url == accepted.url
        return [ChallengeVerdict(url=url, verified=verified, note="")]

    async def _fake_broadened_search(query):
        return [SearchResult(title="넓힌 검색", url=ELEVENST_URL, snippet="12,900원")]

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)
    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)
    monkeypatch.setattr(adk_pipeline_module.search_module, "search", _fake_broadened_search)

    decision = asyncio.run(_relaxed_fallback_decision("무선 마우스 정확한 모델명", []))

    assert decision is not None
    assert decision.url == ELEVENST_URL
    assert decision.verified is True
    assert call_count["pick"] == 2
    assert "검색 범위를 넓혀" in decision.reasoning


def test_relaxed_fallback_decision_marks_unverified_with_caveat_when_challenge_infra_fails(monkeypatch):
    _patch_no_cross_check_signals(monkeypatch)
    verdict = _relaxed_verdict()

    async def _fake_pick(query, search_results):
        return verdict

    async def _fake_challenge(*args, **kwargs):
        return []  # 검증 인프라 장애 시뮬레이션

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)
    monkeypatch.setattr(adk_pipeline_module.deepseek_module, "challenge_candidates", _fake_challenge)

    decision = asyncio.run(_relaxed_fallback_decision("무선 마우스", []))

    assert decision is not None
    assert decision.verified is None
    assert "낮은 확신" in decision.reasoning


def test_relaxed_fallback_decision_none_when_no_candidate_found_at_all(monkeypatch):
    _patch_no_cross_check_signals(monkeypatch)

    async def _fake_pick(query, search_results):
        return None

    monkeypatch.setattr(adk_pipeline_module.gpt_module, "pick_most_relevant", _fake_pick)

    decision = asyncio.run(_relaxed_fallback_decision("마우스", []))

    assert decision is None
