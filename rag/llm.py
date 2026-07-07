"""LLM provider abstraction for the adjudicator.

- MockLLM (default): no network, no secrets. Returns a structured canned
  adjudication derived ONLY from the retrieved clauses, so the RAG path stays
  load-bearing and fully testable offline. Its rationale is explicitly marked
  as mock output.

  It supports two prompt protocols so the whole reasoning/red-team apparatus
  runs offline and deterministically:
    * Direct protocol (default) -- pick the top decisive retrieved clause.
      Behaviour is UNCHANGED from the original service.
    * Chain-of-thought protocol (rag/reasoning.py, marked with COT_SENTINEL in
      the system prompt) -- emit a short, transparent reasoning trace that
      checks the allowed-content exceptions before deciding, then the strict
      JSON verdict. A per-sample `seed` perturbs which exception the path
      notices, so self-consistency (majority vote over sampled paths) is
      meaningful. This is a *model* of an instruction-following LLM for
      offline reproducibility, not a real model -- see reasoning.py.

  The mock is also a deliberately literal instruction follower: in CoT mode it
  honours any imperative it can parse from its prompt (including inside the
  untrusted text region). That makes it a worst-case target for the red-team
  harness (redteam/) -- but note the circularity this buys: the mock obeys
  exactly the imperatives the defenses detect, so the offline before/after ASR
  is a wiring/regression check that is true largely by construction, not
  evidence about any real model's susceptibility (see docs/safety_case.md §4).

- OpenAICompatibleLLM: any OpenAI-compatible chat-completions endpoint,
  selected via environment variables. This covers OpenAI, Azure OpenAI
  (compatible endpoints), vLLM, and Ollama (http://localhost:11434/v1).

Selection (serving startup):
  LLM_BASE_URL + LLM_MODEL set -> OpenAICompatibleLLM (LLM_API_KEY optional,
  e.g. Ollama needs none). Otherwise -> MockLLM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Protocol

logger = logging.getLogger(__name__)

# Marker the CoT prompt builder (rag/reasoning.py) places in the system prompt
# so the offline MockLLM knows to emit a reasoning trace. Real LLMs simply read
# the natural-language instruction; the marker is inert to them.
COT_SENTINEL = "[[REASONING_PROTOCOL]]"


class LLM(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, seed: int | None = None
    ) -> str: ...


# Imperative patterns a naive instruction-follower would obey. Used ONLY by the
# MockLLM to simulate susceptibility so the red-team harness has a worst-case
# target; the production code never executes instructions from input.
_INJECTION_RE = re.compile(
    r"(?:ignore|disregard|forget|override|bypass)\b.{0,60}?"
    r"(?:polic|guideline|instruction|rule|above|previous)",
    re.IGNORECASE | re.DOTALL,
)
_FORCE_LABEL_RE = re.compile(
    r"(?:respond|reply|answer|output|return|label|classify|mark|set|say)\b.{0,40}?"
    r"\b(clean|offensive|hateful)\b",
    re.IGNORECASE | re.DOTALL,
)
_JAILBREAK_FRAMES = (
    "developer mode", "dan mode", "you are now", "this is fiction", "for a novel",
    "hypothetically", "the policy has been updated", "new policy", "without restrictions",
)


class MockLLM:
    """Deterministic offline adjudicator stand-in.

    Direct mode grounds its 'decision' in the retrieved clauses only and is
    byte-for-byte compatible with the original service. CoT mode adds a
    transparent, cue-based reasoning trace (see module docstring).
    """

    name = "mock"

    def complete(self, system: str, user: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        clauses = _extract_clauses(user)
        if COT_SENTINEL in system:
            return self._complete_cot(system, user, clauses, seed)
        return self._complete_direct(clauses)

    # -- direct protocol (UNCHANGED behaviour) ------------------------------- #
    def _complete_direct(self, clauses: list[dict]) -> str:
        decisive = next(
            (c for c in clauses if c.get("category") in ("Hateful", "Offensive", "Clean")), None
        )
        if decisive is None:
            return json.dumps(
                {
                    "final_label": "Clean",
                    "cited_clauses": [],
                    "rationale": "[MOCK] No decisive clause retrieved.",
                }
            )
        return json.dumps(
            {
                "final_label": decisive["category"],
                "cited_clauses": [decisive["clause_id"]],
                "rationale": (
                    f"[MOCK adjudication -- not a real LLM judgment] Top retrieved clause "
                    f"{decisive['clause_id']} ('{decisive.get('title', '')}') is a "
                    f"{decisive['category']} clause; label assigned from it."
                ),
            }
        )

    # -- chain-of-thought protocol ------------------------------------------- #
    def _complete_cot(self, system: str, user: str, clauses: list[dict], seed: int | None) -> str:
        text = _extract_review_text(user)
        available = {c["clause_id"] for c in clauses}

        # A literal instruction-follower: obey any imperative it can parse,
        # UNLESS the hardened system prompt tells it the input is untrusted data
        # AND the untrusted span has been isolated (spotlighting).
        hardened = "UNTRUSTED" in system and "never treat" in system.lower()
        injected = _read_injected_label(text, isolated=hardened)
        if injected is not None:
            label, cid = injected
            cited = [cid] if cid in available else []
            trace = [
                "Step 1 -- I found an explicit instruction in the prompt telling me which label to output.",
                f"Step 2 -- Decision: {label} under {cid}.",
            ]
            return _emit(label, cited, trace, mock=True, hijacked=True)

        label, clause_id, trace, faithful = _reason_over_cues(text, available, seed)
        cited = _ground_citation(label, clause_id, clauses)
        return _emit(label, cited, trace, mock=True, faithful=faithful)


# Masked placeholder cues (never real slurs -- same convention as the rest of
# the project). Real text is handled by a real LLM via LLM_BASE_URL.
def _reason_over_cues(text: str, available: set[str], seed: int | None) -> tuple[str, str, list[str], bool]:
    """Return (label, controlling_clause_id, reasoning_steps, faithful).

    Deterministic given (text, seed). On context-dependent cases an allowed-
    content exception (CL-2/CL-4/CL-5/OF-3) should override the surface reading;
    a per-seed 'miss' models a single reasoning path that overlooks the
    exception, which majority voting over several paths then corrects.
    """
    t = text.lower()
    steps = ["Step 1 -- Read the text and list the policy exceptions that could apply."]

    has_slur = "[slur]" in t
    has_protected = "[protected_group]" in t or "[group]" in t
    condemnation = any(
        p in t
        for p in ("no respect", "people who write", "people who say", "disgusting", "reported", "condemn")
    )
    quoted = any(q in text for q in ('"', "“", "”"))  # double-quote span only (not apostrophes)
    negation = any(
        p in t
        for p in (
            "no group", "nobody deserves", "no one deserves", "does not deserve",
            "deserves to be treated", "it is wrong to", "should never",
        )
    )
    homonym = "[homonym]" in t
    reclaimed = "[reclaimed]" in t
    ingroup = any(p in t for p in ("as one of us", "my friends", "in my community", "in-group"))
    dehum = "[dehum]" in t or "should be removed" in t or "don't belong" in t or "do not belong" in t
    threat = "[threat]" in t
    profanity = "[profanity]" in t
    insult = "[insult]" in t
    generalization = has_protected and any(
        p in t
        for p in ("always", "all of them", "every one of them", "just saying", "just joking", "are all")
    )

    exception = None
    if (quoted or condemnation) and has_slur:
        exception = ("Clean", "CL-2")
    elif negation:
        exception = ("Clean", "CL-4")
    elif homonym:
        exception = ("Clean", "CL-5")
    elif reclaimed and ingroup:
        exception = ("Offensive", "OF-3")

    if dehum and has_protected:
        surface = ("Hateful", "HL-1")
    elif threat and has_protected:
        surface = ("Hateful", "HL-3")
    elif generalization:
        surface = ("Hateful", "HL-5")
    elif has_slur:
        surface = ("Hateful", "HL-2")
    elif reclaimed:
        surface = ("Offensive", "OF-3")
    elif profanity:
        surface = ("Offensive", "OF-1")
    elif insult:
        surface = ("Offensive", "OF-2")
    elif has_protected:
        surface = ("Clean", "CL-3")
    else:
        surface = ("Clean", "CL-1")

    if exception is not None:
        miss = _bucket(text, seed) % 3 == 0  # ~1/3 of sampled paths overlook it
        if miss:
            steps.append(
                f"Step 2 -- The text pattern-matches {surface[1]} on the surface; "
                f"I did not find a clear exception, so I apply the surface reading."
            )
            steps.append(f"Step 3 -- Decision: {surface[0]} under {surface[1]}.")
            return surface[0], surface[1], steps, True
        steps.append(
            f"Step 2 -- An allowed-content exception applies ({exception[1]}); it overrides "
            f"the surface match on {surface[1]}."
        )
        steps.append(f"Step 3 -- Decision: {exception[0]} under {exception[1]}.")
        return exception[0], exception[1], steps, True

    steps.append(f"Step 2 -- No allowed-content exception applies; the text matches {surface[1]}.")
    steps.append(f"Step 3 -- Decision: {surface[0]} under {surface[1]}.")
    return surface[0], surface[1], steps, True


def _ground_citation(label: str, preferred: str, clauses: list[dict]) -> list[str]:
    """A RAG adjudicator may only cite retrieved clauses. Prefer the exact
    clause the reasoning named; otherwise cite the best-ranked retrieved clause
    of the decided category; otherwise cite nothing."""
    ids = [c["clause_id"] for c in clauses]
    if preferred in ids:
        return [preferred]
    same = [c["clause_id"] for c in clauses if c.get("category") == label]
    return same[:1]


def _read_injected_label(review: str, isolated: bool):
    """Simulate susceptibility to instructions embedded in the untrusted text.

    If `isolated` (the text was spotlighted/delimited and the system prompt
    forbids treating it as instructions), the mock ignores imperatives found
    inside the untrusted span -- the defense worked. Otherwise it obeys.
    """
    scan = "" if isolated else review
    frame = any(f in scan.lower() for f in _JAILBREAK_FRAMES)
    forced = _FORCE_LABEL_RE.search(scan)
    overridden = _INJECTION_RE.search(scan)
    # Indirect injection: a clear instruction to emit a label, embedded in the
    # untrusted text, is obeyed (the essence of the vulnerability). A jailbreak
    # frame or override without a named label defaults to the permissive label,
    # which is the attacker's usual goal.
    if forced:
        label = forced.group(1).capitalize()
        clause = {"Clean": "CL-1", "Offensive": "OF-1", "Hateful": "HL-1"}.get(label, "CL-1")
        return label, clause
    if overridden or frame:
        return "Clean", "CL-1"
    return None


def _emit(
    label: str, cited: list[str], steps: list[str], *,
    mock: bool, faithful: bool = True, hijacked: bool = False,
) -> str:
    tag = "[MOCK reasoning -- not a real LLM judgment] " if mock else ""
    rationale = tag + " ".join(steps)
    return json.dumps(
        {
            "final_label": label,
            "cited_clauses": cited,
            "reasoning": steps,
            "rationale": rationale,
            "_mock_meta": {"faithful": faithful, "hijacked": hijacked},
        }
    )


def _extract_clauses(user: str) -> list[dict]:
    try:
        start = user.index("BEGIN_CLAUSES_JSON")
        end = user.index("END_CLAUSES_JSON")
        return json.loads(user[start + len("BEGIN_CLAUSES_JSON"):end])
    except (ValueError, json.JSONDecodeError):
        return []


def _extract_review_text(user: str) -> str:
    """Pull the text-under-review out of the adjudicator's user prompt.
    Works for both the plain and the spotlighted (delimited) prompt formats."""
    m = re.search(r"TEXT UNDER REVIEW:\s*(.*?)\n\s*TIER-1", user, flags=re.DOTALL)
    if not m:
        return ""
    block = m.group(1).strip()
    # Only strip spotlight delimiters when the text is actually wrapped by them
    # (defended path). A stray close-delimiter inside undefended input is left
    # in place, so the red-team "before" numbers are not accidentally defended.
    if block.startswith("<<<UNTRUSTED_INPUT"):
        block = re.sub(r"^<<<UNTRUSTED_INPUT[^\n]*\n?", "", block)
        block = re.sub(r"\n?>>>END_UNTRUSTED_INPUT\s*$", "", block)
    return block.strip()


def _bucket(text: str, seed: int | None) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest(), 16)


class OpenAICompatibleLLM:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0):
        import httpx

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed  # honoured by OpenAI / vLLM; ignored elsewhere
        resp = self._httpx.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_llm() -> LLM:
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if base_url and model:
        logger.info("Using OpenAI-compatible LLM at %s (model=%s).", base_url, model)
        return OpenAICompatibleLLM(base_url, model, api_key=os.environ.get("LLM_API_KEY"))
    logger.warning("LLM_BASE_URL/LLM_MODEL not set -- using MockLLM (canned adjudications).")
    return MockLLM()
