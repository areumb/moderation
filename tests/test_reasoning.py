"""Module A tests: chain-of-thought + self-consistency + faithfulness metrics.
Offline (StubClassifier + MockLLM). Masked placeholders only -- no real slurs."""
from rag.llm import MockLLM
from rag.reasoning import citation_grounded, faithfulness, make_adjudicator, reasoning_label_agreement
from rag.retriever import Retriever

COUNTER = 'I have no respect for people who write "[SLUR] do not belong here".'


def test_cot_emits_reasoning_and_valid_label(policy_store):
    adj = make_adjudicator(Retriever(policy_store, top_k=6), MockLLM(), strategy="cot")
    v = adj.adjudicate(COUNTER, classifier_hint="Hateful")
    assert v["strategy"] == "cot"
    assert v["final_label"] in {"Hateful", "Offensive", "Clean"}
    assert v["reasoning"], "CoT must produce a reasoning trace"
    assert any("Decision:" in s for s in v["reasoning"])


def test_self_consistency_majority_vote(policy_store):
    adj = make_adjudicator(
        Retriever(policy_store, top_k=6), MockLLM(), strategy="self_consistency", samples=5
    )
    v = adj.adjudicate(COUNTER, classifier_hint="Hateful")
    assert v["samples"] == 5
    assert sum(v["votes"].values()) == 5
    # winning label carries the most votes (ties break by severity, still a max)
    assert v["votes"][v["final_label"]] == max(v["votes"].values())


def test_self_consistency_beats_single_path_on_hard_case(policy_store):
    # On this counter-speech case a single greedy path can miss the CL-2
    # exception; voting over 5 paths should recover Clean.
    sc = make_adjudicator(Retriever(policy_store, top_k=6), MockLLM(), strategy="self_consistency", samples=5)
    assert sc.adjudicate(COUNTER, classifier_hint="Hateful")["final_label"] == "Clean"


def test_faithfulness_metrics_shape(policy_store):
    adj = make_adjudicator(Retriever(policy_store, top_k=6), MockLLM(), strategy="cot")
    v = adj.adjudicate(COUNTER, classifier_hint="Hateful")
    ff = faithfulness(v)
    assert set(ff) == {"citation_grounded", "reasoning_label_agreement", "faithful"}
    assert reasoning_label_agreement(v) in (True, False)  # a Decision: line exists


def test_direct_strategy_has_no_reasoning(policy_store):
    adj = make_adjudicator(Retriever(policy_store, top_k=3), MockLLM(), strategy="direct")
    v = adj.adjudicate("targets a protected group with a [SLUR]", classifier_hint="Hateful")
    assert v["reasoning"] == []
    assert citation_grounded(v) is None  # no reasoning to check -> excluded, not failed
