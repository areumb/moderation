"""Tier-2 RAG-grounded adjudicator.

Builds a prompt containing ONLY the text under review and the retrieved
policy clauses, instructs the LLM to decide against the retrieved policy text
(not its own priors), and parses a strict JSON verdict. Citations are
validated against the retrieved clause ids -- the adjudicator cannot cite a
clause it was not shown, which keeps retrieval load-bearing.

The adjudicator is composed of three swappable parts, all defaulting to the
original behaviour so the existing service and tests are unchanged:

  * a reasoning ``strategy`` -- DirectStrategy (default) or the chain-of-thought
    strategies in ``rag/reasoning.py`` (Module A);
  * an optional ``defense`` -- the hardened prompt + input sanitisation +
    verdict integrity check in ``rag/defenses.py`` (Module B);
  * ``samples`` -- >1 turns on self-consistency (majority vote over sampled
    reasoning paths).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from rag.llm import LLM
from rag.retriever import Retriever

logger = logging.getLogger(__name__)

VALID_LABELS = {"Hateful", "Offensive", "Clean"}
_SEVERITY = {"Hateful": 3, "Offensive": 2, "Clean": 1}

SYSTEM_PROMPT = """\
You are a content-moderation adjudicator. You will receive a piece of text and
a set of policy clauses retrieved from the platform's community guidelines.

Decide which single label applies -- "Hateful", "Offensive", or "Clean" --
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


def clauses_json(retrieved: list[dict]) -> str:
    return json.dumps(
        [
            {"clause_id": c["clause_id"], "title": c["title"], "category": c["category"], "text": c["text"]}
            for c in retrieved
        ],
        ensure_ascii=False,
    )


class DirectStrategy:
    """The original single-shot verdict prompt (no explicit reasoning)."""

    name = "direct"
    sampling = False
    system_prompt = SYSTEM_PROMPT

    def build_user(self, text: str, hint: str | None, clauses_json_str: str) -> str:
        return USER_TEMPLATE.format(text=text, hint=hint or "n/a", clauses_json=clauses_json_str)


def parse_verdict(raw: str, retrieved: list[dict]) -> dict:
    """Parse and validate the strict-JSON verdict (shared by all strategies)."""
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
        logger.warning("Adjudicator cited non-retrieved clauses %s -- dropped.", sorted(dropped))

    reasoning = data.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    return {
        "final_label": label,
        "cited_clauses": cited,
        "rationale": str(data.get("rationale", "")).strip() or None,
        "reasoning": reasoning or [],
        "_mock_meta": data.get("_mock_meta"),
    }


class Adjudicator:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        strategy: object | None = None,
        samples: int = 1,
        defense: object | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.strategy = strategy or DirectStrategy()
        self.samples = max(1, int(samples))
        self.defense = defense

    def adjudicate(self, text: str, classifier_hint: str | None = None) -> dict:
        review_text = text
        sanitation = None
        if self.defense is not None:
            review_text, sanitation = self.defense.sanitize(text)

        retrieved = self.retriever.retrieve(review_text, classifier_hint=classifier_hint)
        cj = clauses_json(retrieved)

        system = self.strategy.system_prompt
        body = review_text
        if self.defense is not None:
            system = self.defense.harden(system)
            body = self.defense.wrap(review_text)
        user = self.strategy.build_user(body, classifier_hint, cj)

        if self.samples <= 1 and not getattr(self.strategy, "sampling", False):
            raw = self.llm.complete(system, user)
            verdict = parse_verdict(raw, retrieved)
            verdict["samples"] = 1
            verdict["votes"] = {verdict["final_label"]: 1}
        else:
            verdict = self._self_consistency(system, user, retrieved)

        if self.defense is not None:
            verdict["sanitation"] = sanitation
            verdict["integrity"] = self.defense.check(verdict, retrieved)
            if not verdict["integrity"]["consistent"]:
                # Defense-in-depth: an internally inconsistent verdict (e.g. a
                # "Clean" that cites only Hateful clauses) is not trusted.
                verdict["final_label"] = verdict["integrity"]["fallback_label"]
                verdict["integrity"]["overridden"] = True

        verdict["retrieved"] = retrieved
        verdict["strategy"] = getattr(self.strategy, "name", "direct")
        return verdict

    def _self_consistency(self, system: str, user: str, retrieved: list[dict]) -> dict:
        """Sample several reasoning paths and take the majority label.

        Wang et al. (2022): sampling multiple chains and voting is more robust
        than a single greedy chain, especially on the context-dependent cases
        the thesis found unreliable. Ties break toward the more severe label.
        """
        paths = []
        for i in range(self.samples):
            raw = self.llm.complete(system, user, temperature=0.7, seed=i)
            try:
                paths.append(parse_verdict(raw, retrieved))
            except AdjudicationError:
                continue
        if not paths:
            raise AdjudicationError("no reasoning path produced a valid verdict")

        votes = Counter(p["final_label"] for p in paths)
        top = max(votes.items(), key=lambda kv: (kv[1], _SEVERITY[kv[0]]))[0]
        winners = [p for p in paths if p["final_label"] == top]
        chosen = winners[0]
        cited = []
        for p in winners:
            for c in p["cited_clauses"]:
                if c not in cited:
                    cited.append(c)
        return {
            "final_label": top,
            "cited_clauses": cited,
            "rationale": chosen["rationale"],
            "reasoning": chosen["reasoning"],
            "_mock_meta": chosen.get("_mock_meta"),
            "samples": len(paths),
            "votes": dict(votes),
            "paths": [
                {
                    "final_label": p["final_label"],
                    "cited_clauses": p["cited_clauses"],
                    "reasoning": p["reasoning"],
                }
                for p in paths
            ],
        }
