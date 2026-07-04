"""Routing between Tier 1 (classifier) and Tier 2 (RAG adjudicator).

The escalation criteria come directly from the thesis finding: the
clean-vs-not-clean distinction is comparatively easy, while the
Offensive <-> Hateful boundary is where the classifier is least reliable.
So we escalate exactly there:

  (a) top label is Offensive or Hateful (configurable),
  (b) confidence below `conf_threshold`,
  (c) |P(Offensive) - P(Hateful)| below `margin_threshold`.

Anything else (confident Clean) is auto-resolved by Tier 1 — except for a
deterministic audit sample (see `should_audit`).

Why there is no explicit Clean<->Hateful margin rule
----------------------------------------------------
The thesis' HateCheck-XR results show the hard boundary shifts out of
distribution: hateful cases are frequently misclassified as Clean. A
"|P(Clean) - P(Hateful)| < margin" rule looks like the obvious response, but
it is mathematically subsumed by the confidence rule: if the top label is
Clean with confidence c >= tau, then P(Hateful) <= 1 - c, so the margin is at
least 2*tau - 1 (0.40 at tau = 0.70). Such a rule could therefore only fire
when c < (1 + margin)/2 — i.e. on inputs the confidence rule already
escalates for any tau >= 0.575. The residual risk is *confident*
Hateful->Clean errors, which no probability-based rule can catch; that is
what audit sampling below is for, and why the behavioral suite tracks the
gold-Hateful->Clean rate explicitly (evals/run_behavioral_suite.py).

The reverse OOD error the thesis also found sizable — benign counterspeech,
quotation, negation, or slur homonyms misread as Hateful (Clean->Hate) —
needs no extra rule either: any non-Clean top label already escalates, and
the adjudicator settles it against the retrieved allowed-content clauses
(CL-2 counterspeech, CL-4 negation, CL-5 homonyms). The behavioral suite
tracks that direction too, as over-moderation has its own cost.
"""
from __future__ import annotations

import hashlib

from serving.config import ServingConfig


def should_escalate(classifier_output: dict, cfg: ServingConfig) -> tuple[bool, list[str]]:
    """Return (escalate?, reasons). Reasons are included in the API response
    so every routing decision is auditable."""
    reasons: list[str] = []
    label = classifier_output["label"]
    probs = classifier_output["probs"]
    confidence = classifier_output["confidence"]

    if label in cfg.escalate_labels:
        reasons.append(f"top_label:{label}")
    if confidence < cfg.conf_threshold:
        reasons.append(f"low_confidence:{confidence:.3f}<{cfg.conf_threshold}")
    p_off = probs.get("Offensive", 0.0)
    p_hate = probs.get("Hateful", 0.0)
    margin = abs(p_off - p_hate)
    # Margin rule: only meaningful when Offensive/Hateful carry real mass —
    # for confident Clean both are near zero and thus always "close".
    if max(p_off, p_hate) >= cfg.margin_applies_above and margin < cfg.margin_threshold:
        reasons.append(f"offensive_hateful_margin:{margin:.3f}<{cfg.margin_threshold}")

    return (len(reasons) > 0, reasons)


def should_audit(text: str, cfg: ServingConfig) -> bool:
    """Deterministic audit sampling of the auto-approved bucket.

    Motivated directly by the thesis finding that the classifier over-relies
    on overt lexical cues and confidently mislabels implicit/coded hate as
    Clean: those items sail past every probability-based trigger, so a sample
    of auto-approved traffic is sent to the Tier-2 adjudicator anyway. This is
    the standard trust & safety mitigation for the confident-miss blind spot;
    it reduces the exposure (and measures the miss rate on the sample), it
    does not eliminate it.

    Sampling is a pure function of the text (SHA-256 bucket), not an RNG:
    the same input always gets the same treatment, which makes the behavior
    reproducible in tests and audits and idempotent across retries.
    """
    if cfg.audit_rate <= 0.0:
        return False
    if cfg.audit_rate >= 1.0:
        return True
    bucket = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 1_000_000
    return bucket < int(cfg.audit_rate * 1_000_000)
