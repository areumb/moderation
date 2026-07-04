"""Router unit tests — every escalation trigger, plus the auto path.

All fixtures use masked placeholders; no real slurs (project policy).
"""
from serving.config import ServingConfig
from serving.router import should_audit, should_escalate

CFG = ServingConfig(conf_threshold=0.70, margin_threshold=0.15)


def _clf(hate, off, clean):
    probs = {"Hateful": hate, "Offensive": off, "Clean": clean}
    label = max(probs, key=probs.get)
    return {"label": label, "probs": probs, "confidence": probs[label]}


def test_confident_clean_is_auto():
    escalate, reasons = should_escalate(_clf(0.02, 0.05, 0.93), CFG)
    assert not escalate
    assert reasons == []


def test_top_label_hateful_escalates():
    escalate, reasons = should_escalate(_clf(0.90, 0.07, 0.03), CFG)
    assert escalate
    assert any(r.startswith("top_label:Hateful") for r in reasons)


def test_top_label_offensive_escalates():
    escalate, reasons = should_escalate(_clf(0.08, 0.82, 0.10), CFG)
    assert escalate
    assert any(r.startswith("top_label:Offensive") for r in reasons)


def test_low_confidence_clean_escalates():
    escalate, reasons = should_escalate(_clf(0.20, 0.25, 0.55), CFG)
    assert escalate
    assert any(r.startswith("low_confidence") for r in reasons)


def test_offensive_hateful_margin_escalates():
    # Confident enough overall, but Offensive and Hateful nearly tied —
    # exactly the boundary the thesis found unreliable.
    out = _clf(0.44, 0.46, 0.10)
    escalate, reasons = should_escalate(out, CFG)
    assert escalate
    assert any(r.startswith("offensive_hateful_margin") for r in reasons)


def test_thresholds_configurable():
    lax = ServingConfig(conf_threshold=0.0, margin_threshold=0.0, escalate_labels=[])
    escalate, _ = should_escalate(_clf(0.44, 0.46, 0.10), lax)
    assert not escalate


# --- Audit sampling (blind-spot mitigation for confident Hateful->Clean) ---


def test_audit_rate_zero_never_samples():
    cfg = ServingConfig(audit_rate=0.0)
    assert not any(should_audit(f"text {i}", cfg) for i in range(100))


def test_audit_rate_one_always_samples():
    cfg = ServingConfig(audit_rate=1.0)
    assert all(should_audit(f"text {i}", cfg) for i in range(100))


def test_audit_sampling_is_deterministic_per_text():
    cfg = ServingConfig(audit_rate=0.3)
    texts = [f"sample text {i}" for i in range(200)]
    first = [should_audit(t, cfg) for t in texts]
    second = [should_audit(t, cfg) for t in texts]
    assert first == second


def test_audit_sampling_rate_roughly_respected():
    # SHA-256 buckets are ~uniform; with n=400 and p=0.5 the count is within
    # these (very loose, ±6σ) bounds for any uniform hash. Deterministic.
    cfg = ServingConfig(audit_rate=0.5)
    sampled = sum(should_audit(f"case {i}", cfg) for i in range(400))
    assert 140 <= sampled <= 260
