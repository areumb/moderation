"""RAG-path tests: policy parsing, retrieval, adjudication with MockLLM."""
import json

import pytest

from rag.adjudicator import AdjudicationError, Adjudicator
from rag.llm import MockLLM
from rag.retriever import Retriever


def test_policy_parses_into_clauses_with_stable_ids(policy_store):
    ids = [c.clause_id for c in policy_store.clauses]
    assert len(ids) == len(set(ids))
    assert any(i.startswith("HL-") for i in ids)
    assert any(i.startswith("OF-") for i in ids)
    assert any(i.startswith("CL-") for i in ids)
    categories = {c.category for c in policy_store.clauses}
    assert {"Hateful", "Offensive", "Clean"} <= categories


def test_index_build_is_idempotent(policy_store):
    count_before = policy_store.collection.count()
    rebuilt = policy_store._build_if_needed()  # second call must not duplicate
    assert rebuilt.count() == count_before == len(policy_store.clauses)


def test_retrieval_returns_clauses(policy_store):
    retriever = Retriever(policy_store, top_k=3)
    results = retriever.retrieve("this message insults a protected group with a [SLUR]")
    assert len(results) == 3
    for r in results:
        assert r["clause_id"] in policy_store.by_id
        assert r["text"]


def test_adjudicator_parses_mock_verdict(policy_store):
    adj = Adjudicator(Retriever(policy_store, top_k=3), MockLLM())
    verdict = adj.adjudicate("targets a protected group with a [SLUR]", classifier_hint="Hateful")
    assert verdict["final_label"] in {"Hateful", "Offensive", "Clean"}
    assert verdict["cited_clauses"], "MockLLM must cite a retrieved clause"
    retrieved_ids = {c["clause_id"] for c in verdict["retrieved"]}
    assert set(verdict["cited_clauses"]) <= retrieved_ids, "citations must come from retrieval"
    assert "[MOCK" in verdict["rationale"]


def test_adjudicator_rejects_garbage_llm_output(policy_store):
    class BrokenLLM:
        name = "broken"

        def complete(self, system, user):
            return "not json at all"

    adj = Adjudicator(Retriever(policy_store, top_k=2), BrokenLLM())
    with pytest.raises(AdjudicationError):
        adj.adjudicate("anything")


def test_adjudicator_drops_non_retrieved_citations(policy_store):
    class HallucinatingLLM:
        name = "hallucinating"

        def complete(self, system, user):
            return json.dumps(
                {"final_label": "Hateful", "cited_clauses": ["FAKE-99"], "rationale": "made up"}
            )

    adj = Adjudicator(Retriever(policy_store, top_k=2), HallucinatingLLM())
    verdict = adj.adjudicate("some text")
    assert verdict["cited_clauses"] == []  # hallucinated citation removed
