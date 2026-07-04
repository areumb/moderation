"""Tier-2 RAG-grounded adjudicator.

Builds a prompt containing ONLY the text under review and the retrieved
policy clauses, instructs the LLM to decide against the retrieved policy text
(not its own priors), and parses a strict JSON verdict. Citations are
validated against the retrieved clause ids — the adjudicator cannot cite a
clause it was not shown, which keeps retrieval load-bearing.
"""
from __future__ import annotations

import json
import logging
import re

from rag.llm import LLM
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

VALID_LABELS = {"Hateful", "Offensive", "Clean"}

SYSTEM_PROMPT = """\
You are a content-moderation adjudicator. You will receive a piece of text and
a set of policy clauses retrieved from the platform's community guidelines.

Decide which single label applies — "Hateful", "Offensive", or "Clean" —
STRICTLY according to the retrieved clauses. The retrieved policy text is the
only ground truth: do not rely on your own definitions of these terms, and do
not cite any clause that is not in the provided set. If the retrieved clauses
genuinely do not cover the text, choose the closest applicable clause and say
so in the rationale.

Respond with ONLY a JSON object, no prose around it:
{"final_label": "<Hateful|Offensive|Clean>",
 "cited_clauses": ["<clause_id>", ...],
 "rationale": "<one or two sentences grounded in the cited clauses>"}
"""

USER_TEMPLATE = """\
TEXT UNDER REVIEW:
{text}

TIER-1 CLASSIFIER (tentative, may be wrong): {hint}

RETRIEVED POLICY CLAUSES:
BEGIN_CLAUSES_JSON{clauses_json}END_CLAUSES_JSON
"""


class AdjudicationError(Exception):
    pass


class Adjudicator:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def adjudicate(self, text: str, classifier_hint: str | None = None) -> dict:
        retrieved = self.retriever.retrieve(text, classifier_hint=classifier_hint)
        clauses_json = json.dumps(
            [
                {
                    "clause_id": c["clause_id"],
                    "title": c["title"],
                    "category": c["category"],
                    "text": c["text"],
                }
                for c in retrieved
            ],
            ensure_ascii=False,
        )
        user = USER_TEMPLATE.format(text=text, hint=classifier_hint or "n/a", clauses_json=clauses_json)
        raw = self.llm.complete(SYSTEM_PROMPT, user)
        verdict = self._parse(raw, retrieved)
        verdict["retrieved"] = retrieved
        return verdict

    def _parse(self, raw: str, retrieved: list[dict]) -> dict:
        """Parse and validate the strict-JSON verdict."""
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise AdjudicationError(f"LLM returned no JSON object: {raw[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise AdjudicationError(f"LLM returned invalid JSON: {exc}") from exc

        label = data.get("final_label")
        if label not in VALID_LABELS:
            raise AdjudicationError(f"Invalid final_label: {label!r}")

        retrieved_ids = {c["clause_id"] for c in retrieved}
        cited = [c for c in data.get("cited_clauses", []) if c in retrieved_ids]
        dropped = set(data.get("cited_clauses", [])) - retrieved_ids
        if dropped:
            logger.warning("Adjudicator cited non-retrieved clauses %s — dropped.", sorted(dropped))

        return {
            "final_label": label,
            "cited_clauses": cited,
            "rationale": str(data.get("rationale", "")).strip() or None,
        }
