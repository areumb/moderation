"""Pydantic schemas for the moderation API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModerateRequest(BaseModel):
    text: str = Field(..., description="Text to moderate.")
    mode: Literal["3class", "hate_nonhate", "nonclean_clean"] = Field(
        "3class",
        description=(
            "Optional label-space view. Binary modes add a projection of the "
            "ternary probabilities (same merging rule as the thesis code); "
            "routing and adjudication always run on the ternary output."
        ),
    )


class ClassifierOutput(BaseModel):
    label: str
    probs: dict[str, float]
    confidence: float
    projected: Optional[dict] = None
    engine: str = Field(description="'hf' for the real fine-tuned model, 'stub' for the offline stub.")
    # Module B (tier1_evasion defense): transforms that changed the classifier's
    # input when NORMALIZE_TIER1 is on (strip_zero_width / homoglyph_fold /
    # de_leet / despace). A non-empty list is itself a cheap obfuscation signal.
    tier1_normalization: list[str] = []


class PolicyCitation(BaseModel):
    clause_id: str
    text: str


class Decision(BaseModel):
    final_label: str
    # auto: Tier 1 accepted as-is. escalated: a router trigger fired.
    # audit: no trigger fired, but the item was deterministically sampled from
    # the auto-approved bucket and adjudicated anyway (blind-spot mitigation).
    route: Literal["auto", "escalated", "audit"]
    escalation_reasons: list[str] = []
    policy_citations: list[PolicyCitation] = []
    rationale: Optional[str] = None
    adjudicator: Optional[str] = Field(
        None, description="'mock' or the configured LLM model name; null on the auto route."
    )
    # Module A: chain-of-thought reasoning surfaced for auditability.
    strategy: Optional[str] = None
    reasoning: list[str] = []
    votes: Optional[dict] = None
    # Module B: verdict integrity check + detected attack markers (hardened mode).
    integrity: Optional[dict] = None
    attack_markers: list[str] = []


class ModerateResponse(BaseModel):
    text: str
    classifier: ClassifierOutput
    decision: Decision


class HealthResponse(BaseModel):
    status: str
    classifier_engine: str
    llm_provider: str
    policy_clauses: int


class StatsResponse(BaseModel):
    """Routing statistics since process start (in-memory, reset on restart).

    tier2_rate is the operative cost/quality number: the share of traffic that
    actually reached the expensive adjudicator. The same thresholds produce
    very different rates on different traffic mixes (e.g. Davidson is ~77%
    Offensive; production traffic is mostly Clean), so this must be observed,
    not assumed.
    """

    total: int
    routes: dict[str, int]
    tier2_rate: float
    trigger_counts: dict[str, int]
    note: str = "In-memory counters since process start; reset on restart."
