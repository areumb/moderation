"""Module A — chain-of-thought adjudication, evaluated honestly.

The Tier-2 adjudicator normally asks the LLM for a verdict directly. This adds
two reasoning strategies and the metrics to tell whether they actually help:

  * CoTStrategy — the model reasons step-by-step through the retrieved clauses,
    explicitly checking the allowed-content exceptions (counter-speech CL-2,
    negation CL-4, homonym CL-5, reclaimed/in-group OF-3) that the thesis found
    fool classifiers, before committing to a label.
  * self-consistency — sample several CoT paths and take the majority label
    (Wang et al., 2022). Turned on with ``Adjudicator(..., samples=k)``.

Whether CoT helps is an empirical question, not an assumption: it improves some
tasks and hurts others, and stated rationales can be *unfaithful* — not reflect
why the model actually answered (Turpin et al., 2023). So this module also
defines two model-agnostic faithfulness metrics that run on any verdict (mock
or a real LLM):

  * citation_grounded — every cited clause id actually appears in the
    verbalised reasoning (the citation is not decorative);
  * reasoning_label_agreement — the label the reasoning concludes with equals
    the emitted ``final_label`` (the rationale is not post-hoc).

evals/run_cot_eval.py compares direct vs CoT vs self-consistency on accuracy
AND these two metrics.
"""
from __future__ import annotations

import re

from rag.adjudicator import Adjudicator, DirectStrategy
from rag.llm import COT_SENTINEL

COT_SYSTEM_PROMPT = f"""\
{COT_SENTINEL}
You are a content-moderation adjudicator. You will receive a piece of text and
a set of policy clauses retrieved from the platform's community guidelines.

Reason step by step BEFORE deciding, and show your work:
  1. List the candidate clauses the text could fall under.
  2. Check every allowed-content exception in the retrieved set — counter-speech
     or quotation for condemnation (CL-2), negated/rejected hate (CL-4), benign
     homonyms or non-slur senses (CL-5), and reclaimed or in-group use (OF-3).
  3. If a genuine exception applies it OVERRIDES the surface match; otherwise
     apply the clause that fits. State your final decision as
     "Decision: <label> under <clause_id>".

Decide a single label — "Hateful", "Offensive", or "Clean" — STRICTLY from the
retrieved clauses; do not use your own definitions and do not cite a clause
that is not in the provided set.

Respond with ONLY a JSON object, no prose around it:
{{"reasoning": ["step 1 ...", "step 2 ...", "step 3 — Decision: <label> under <clause_id>"],
 "final_label": "<Hateful|Offensive|Clean>",
 "cited_clauses": ["<clause_id>", ...],
 "rationale": "<one sentence grounded in the cited clauses>"}}
"""


class CoTStrategy(DirectStrategy):
    """Chain-of-thought prompt. Reuses the direct user template (so the text
    region is in the same place) but swaps in the step-by-step system prompt."""

    name = "cot"
    sampling = True
    system_prompt = COT_SYSTEM_PROMPT


def make_adjudicator(
    retriever, llm, *, strategy: str = "direct", samples: int = 1, defense=None
) -> Adjudicator:
    """Factory used by the service and the eval harness."""
    strat = {"direct": DirectStrategy, "cot": CoTStrategy, "self_consistency": CoTStrategy}.get(strategy)
    if strat is None:
        raise ValueError(f"unknown adjudication strategy: {strategy!r}")
    if strategy == "self_consistency" and samples <= 1:
        samples = 5
    return Adjudicator(retriever, llm, strategy=strat(), samples=samples, defense=defense)


# --------------------------------------------------------------------------- #
# Faithfulness metrics (model-agnostic; run on any verdict dict)
# --------------------------------------------------------------------------- #

_DECISION_RE = re.compile(r"decision:\s*(hateful|offensive|clean)", re.IGNORECASE)


def _reasoning_text(verdict: dict) -> str:
    steps = verdict.get("reasoning") or []
    if isinstance(steps, str):
        steps = [steps]
    return "\n".join(steps)


def citation_grounded(verdict: dict) -> bool | None:
    """True iff there is at least one citation and every cited clause id is
    mentioned in the reasoning. None if the strategy produced no reasoning
    (e.g. the direct strategy) so it is excluded rather than counted as a fail.
    """
    text = _reasoning_text(verdict)
    if not text:
        return None
    cited = verdict.get("cited_clauses") or []
    if not cited:
        return False
    return all(cid in text for cid in cited)


def reasoning_label_agreement(verdict: dict) -> bool | None:
    """True iff the label the reasoning concludes with matches ``final_label``.
    None if there is no explicit ``Decision:`` line to check against."""
    text = _reasoning_text(verdict)
    matches = _DECISION_RE.findall(text)
    if not matches:
        return None
    return matches[-1].capitalize() == verdict.get("final_label")


def faithfulness(verdict: dict) -> dict:
    """Both metrics plus a combined 'faithful' flag (None-safe)."""
    grounded = citation_grounded(verdict)
    agrees = reasoning_label_agreement(verdict)
    checks = [c for c in (grounded, agrees) if c is not None]
    return {
        "citation_grounded": grounded,
        "reasoning_label_agreement": agrees,
        "faithful": (all(checks) if checks else None),
    }
