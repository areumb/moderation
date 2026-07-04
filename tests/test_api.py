"""API tests (offline: StubClassifier + MockLLM).

Test inputs use masked placeholders like "[SLUR]" — no real slurs.
"""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["classifier_engine"] == "stub"
    assert body["llm_provider"] == "mock"
    assert body["policy_clauses"] > 0


def test_clean_text_auto_route(client):
    resp = client.post("/moderate", json={"text": "I really enjoyed the community picnic today."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["classifier"]["label"] == "Clean"
    assert set(body["classifier"]["probs"]) == {"Hateful", "Offensive", "Clean"}
    assert body["decision"]["route"] == "auto"
    assert body["decision"]["final_label"] == "Clean"
    assert body["decision"]["policy_citations"] == []
    assert body["decision"]["rationale"] is None


def test_masked_hateful_text_escalates_with_citations(client):
    resp = client.post("/moderate", json={"text": "people like that are [SLUR] and don't belong here"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["classifier"]["label"] == "Hateful"
    d = body["decision"]
    assert d["route"] == "escalated"
    assert d["escalation_reasons"]
    assert d["policy_citations"], "escalated decision must carry policy citations"
    for citation in d["policy_citations"]:
        assert citation["clause_id"]
        assert citation["text"]
    assert d["final_label"] in {"Hateful", "Offensive", "Clean"}
    assert d["rationale"]


def test_ambiguous_text_escalates_on_margin(client):
    resp = client.post("/moderate", json={"text": "that [AMBIGUOUS] remark again"})
    body = resp.json()
    assert body["decision"]["route"] == "escalated"
    assert any(r.startswith("offensive_hateful_margin") for r in body["decision"]["escalation_reasons"])


def test_binary_projection_view(client):
    resp = client.post(
        "/moderate",
        json={"text": "people like that are [SLUR] and don't belong here", "mode": "hate_nonhate"},
    )
    body = resp.json()
    proj = body["classifier"]["projected"]
    assert proj["mode"] == "hate_nonhate"
    assert proj["label"] == "Hateful"
    assert abs(sum(proj["probs"].values()) - 1.0) < 1e-4


def test_empty_input_rejected(client):
    assert client.post("/moderate", json={"text": "   "}).status_code == 422


def test_oversized_input_rejected(client):
    assert client.post("/moderate", json={"text": "x" * 20_001}).status_code == 422


def test_missing_field_rejected(client):
    assert client.post("/moderate", json={}).status_code == 422


def test_stats_endpoint_counts_routes(client):
    before = client.get("/stats").json()
    client.post("/moderate", json={"text": "lovely weather for the neighborhood picnic"})
    client.post("/moderate", json={"text": "people like that are [SLUR] and don't belong here"})
    after = client.get("/stats").json()
    assert after["total"] == before["total"] + 2
    assert after["routes"].get("auto", 0) == before["routes"].get("auto", 0) + 1
    assert after["routes"].get("escalated", 0) == before["routes"].get("escalated", 0) + 1
    assert 0.0 <= after["tier2_rate"] <= 1.0
    assert after["trigger_counts"].get("top_label", 0) >= 1


def test_audit_route_sends_sampled_clean_to_tier2(offline_env, monkeypatch):
    """With AUDIT_RATE=1.0 every auto-approved item is audited: a confident
    Clean prediction still reaches Tier 2 and returns policy citations."""
    monkeypatch.setenv("AUDIT_RATE", "1.0")
    from fastapi.testclient import TestClient

    from serving.app import app

    with TestClient(app) as audit_client:
        resp = audit_client.post(
            "/moderate", json={"text": "I really enjoyed the community picnic today."}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["classifier"]["label"] == "Clean"
        d = body["decision"]
        assert d["route"] == "audit"
        assert d["escalation_reasons"] == ["audit_sample"]
        assert d["policy_citations"], "audited decision must be policy-grounded like any Tier-2 verdict"
        stats = audit_client.get("/stats").json()
        assert stats["routes"].get("audit", 0) == 1
        assert stats["trigger_counts"].get("audit_sample", 0) == 1
