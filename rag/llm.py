"""LLM provider abstraction for the adjudicator.

- MockLLM (default): no network, no secrets. Returns a structured canned
  adjudication derived ONLY from the retrieved clauses, so the RAG path stays
  load-bearing and fully testable offline. Its rationale is explicitly marked
  as mock output.
- OpenAICompatibleLLM: any OpenAI-compatible chat-completions endpoint,
  selected via environment variables. This covers OpenAI, Azure OpenAI
  (compatible endpoints), vLLM, and Ollama (http://localhost:11434/v1).

Selection (serving startup):
  LLM_BASE_URL + LLM_MODEL set -> OpenAICompatibleLLM (LLM_API_KEY optional,
  e.g. Ollama needs none). Otherwise -> MockLLM.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class MockLLM:
    """Deterministic offline adjudicator stand-in.

    It grounds its 'decision' in the retrieved clauses only: the user prompt
    built by the adjudicator embeds the retrieved clauses as JSON; the mock
    picks the highest-ranked non-definition clause and answers with that
    clause's category and id. No real reasoning happens — it exists so the
    full request path (retrieve -> prompt -> parse -> respond) runs in CI.
    """

    name = "mock"

    def complete(self, system: str, user: str) -> str:
        clauses = []
        try:
            start = user.index("BEGIN_CLAUSES_JSON")
            end = user.index("END_CLAUSES_JSON")
            clauses = json.loads(user[start + len("BEGIN_CLAUSES_JSON"):end])
        except (ValueError, json.JSONDecodeError):
            pass

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
                    f"[MOCK adjudication — not a real LLM judgment] Top retrieved clause "
                    f"{decisive['clause_id']} ('{decisive.get('title', '')}') is a "
                    f"{decisive['category']} clause; label assigned from it."
                ),
            }
        )


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

    def complete(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
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
    logger.warning("LLM_BASE_URL/LLM_MODEL not set — using MockLLM (canned adjudications).")
    return MockLLM()
