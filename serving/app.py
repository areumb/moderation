"""Tiered, policy-grounded moderation API.

    text -> Tier 1 classifier -> router -> [auto]      -> final label
                                        -> [escalated] -> RAG adjudicator -> final label + citations

Run locally (offline, stub classifier + mock LLM):
    uvicorn serving.app:app --reload

Activate the real path with environment variables only (no code changes):
    MODEL_DIR=outputs/davidson/RoBERTa-base/3class  (thesis .pt checkpoint dir)
    LLM_BASE_URL=... LLM_MODEL=... [LLM_API_KEY=...]
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rag.adjudicator import AdjudicationError
from rag.defenses import PromptDefense, normalize_for_classifier
from rag.llm import get_llm
from rag.policy_store import PolicyStore
from rag.reasoning import make_adjudicator
from rag.retriever import Retriever
from serving.config import ServingConfig
from serving.predictor import get_classifier, project
from serving.router import should_audit, should_escalate
from serving.schemas import (
    ClassifierOutput,
    Decision,
    HealthResponse,
    ModerateRequest,
    ModerateResponse,
    PolicyCitation,
    StatsResponse,
)
from serving.stats import RouteStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ServingConfig.load()
    store = PolicyStore(cfg.policy_path, cfg.chroma_dir, cfg.collection_name)
    llm = get_llm()
    state["cfg"] = cfg
    state["classifier"] = get_classifier()
    state["store"] = store
    defense = PromptDefense() if cfg.harden_adjudicator else None
    state["adjudicator"] = make_adjudicator(
        Retriever(store, top_k=cfg.top_k),
        llm,
        strategy=cfg.adjudication_strategy,
        samples=cfg.self_consistency_samples,
        defense=defense,
    )
    state["llm_name"] = llm.name
    state["stats"] = RouteStats()
    yield
    state.clear()


app = FastAPI(
    title="Tiered Policy-Grounded Moderation Service",
    description="Fast classifier triage + RAG policy adjudication for the hard Offensive/Hateful tail.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        classifier_engine=state["classifier"].name,
        llm_provider=state["llm_name"],
        policy_clauses=len(state["store"].clauses),
    )


@app.post("/moderate", response_model=ModerateResponse)
def moderate(req: ModerateRequest) -> ModerateResponse:
    cfg: ServingConfig = state["cfg"]

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(text) > cfg.max_input_chars:
        raise HTTPException(status_code=422, detail=f"text exceeds {cfg.max_input_chars} characters")

    # Tier 1 — optionally canonicalise obfuscation (spacing/leet/homoglyphs)
    # before lexical classification (Module B, tier1_evasion defense). Only the
    # classifier sees the normalised form; the original text is what is stored,
    # adjudicated, and returned.
    if cfg.normalize_tier1:
        clf_text, normalization_applied = normalize_for_classifier(text)
    else:
        clf_text, normalization_applied = text, []
    clf_out = state["classifier"].predict(clf_text)
    projected = project(clf_out["probs"], req.mode) if req.mode != "3class" else None

    # Router: risk triggers first; otherwise a deterministic audit sample of
    # the auto-approved bucket still goes to Tier 2 (the confident
    # Hateful->Clean blind spot cannot be caught by probability thresholds).
    escalate, reasons = should_escalate(clf_out, cfg)
    if escalate:
        route = "escalated"
    elif should_audit(text, cfg):
        route, reasons = "audit", ["audit_sample"]
    else:
        route = "auto"

    if route == "auto":
        decision = Decision(final_label=clf_out["label"], route="auto", escalation_reasons=[])
    else:
        # Tier 2 — RAG-grounded adjudication. Audited items are adjudicated
        # exactly like escalations and the verdict stands; a log-only QA
        # variant would keep Tier 1's label instead.
        try:
            verdict = state["adjudicator"].adjudicate(text, classifier_hint=clf_out["label"])
        except AdjudicationError as exc:
            logger.error("Adjudication failed: %s — falling back to classifier label.", exc)
            decision = Decision(
                final_label=clf_out["label"],
                route=route,
                escalation_reasons=reasons,
                rationale=f"Adjudicator unavailable ({exc}); classifier label used as fallback.",
                adjudicator=state["llm_name"],
            )
        else:
            by_id = {c["clause_id"]: c for c in verdict["retrieved"]}
            integrity = verdict.get("integrity")
            decision = Decision(
                final_label=verdict["final_label"],
                route=route,
                escalation_reasons=reasons,
                policy_citations=[
                    PolicyCitation(clause_id=cid, text=by_id[cid]["text"]) for cid in verdict["cited_clauses"]
                ],
                rationale=verdict["rationale"],
                adjudicator=state["llm_name"],
                strategy=verdict.get("strategy"),
                reasoning=verdict.get("reasoning", []),
                votes=verdict.get("votes"),
                integrity=integrity,
                attack_markers=(integrity or {}).get("attack_markers", []),
            )

    state["stats"].record(decision.route, decision.escalation_reasons)
    return ModerateResponse(
        text=req.text,
        classifier=ClassifierOutput(
            **clf_out,
            projected=projected,
            engine=state["classifier"].name,
            tier1_normalization=normalization_applied,
        ),
        decision=decision,
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Observed routing distribution — how much traffic Tier 2 actually costs."""
    return StatsResponse(**state["stats"].snapshot())
