"""Serving configuration.

Thresholds and RAG settings are loaded from a JSON config file
(default: serving/config.json, override with the SERVING_CONFIG env var).
Individual values can also be overridden via environment variables.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"

# Canonical label names, aligned with the base encoding used throughout
# hs_generalization (0=hate, 1=offensive, 2=clean; see hs_generalization/modes.py).
LABELS = ["Hateful", "Offensive", "Clean"]


@dataclass
class ServingConfig:
    # Router thresholds
    conf_threshold: float = 0.70
    margin_threshold: float = 0.15
    # The Offensive/Hateful margin rule only applies when at least one of the
    # two has meaningful mass; otherwise confident-Clean outputs (where both
    # are tiny and therefore always close) would trivially escalate.
    margin_applies_above: float = 0.20
    escalate_labels: list[str] = field(default_factory=lambda: ["Hateful", "Offensive"])
    # Deterministic audit sampling of the auto-approved bucket (0.0-1.0).
    # Mitigates the confident Hateful->Clean blind spot found on HateCheck-XR;
    # see serving/router.py:should_audit.
    audit_rate: float = 0.02

    # Tier-2 adjudication strategy (Module A) + hardening (Module B).
    adjudication_strategy: str = "direct"   # direct | cot | self_consistency
    self_consistency_samples: int = 5
    harden_adjudicator: bool = False
    # Tier-1 evasion defense (Module B): canonicalise obfuscation (spacing /
    # leetspeak / homoglyphs) before lexical classification. Applied to the
    # classifier's INPUT only — the original text is what is stored,
    # adjudicated, and returned. See rag/defenses.py:normalize_for_classifier.
    normalize_tier1: bool = False

    # RAG settings. top_k=6 so the allowed-content exception clauses
    # (CL-2/CL-4/CL-5, OF-3) are reliably retrieved alongside the surface-
    # matching clauses: the CoT strategy is instructed to check them, and the
    # hardened integrity check only accepts an exception it can see. This also
    # matches the default the evals/redteam harnesses are characterised at.
    top_k: int = 6
    policy_path: str = "policies/community_guidelines.md"
    chroma_dir: str = ".chroma"
    collection_name: str = "community_guidelines"

    # Input validation
    max_input_chars: int = 10_000

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "ServingConfig":
        path = Path(path or os.environ.get("SERVING_CONFIG", DEFAULT_CONFIG_PATH))
        data = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        # Environment overrides (useful in Docker / CI).
        if "CONF_THRESHOLD" in os.environ:
            cfg.conf_threshold = float(os.environ["CONF_THRESHOLD"])
        if "MARGIN_THRESHOLD" in os.environ:
            cfg.margin_threshold = float(os.environ["MARGIN_THRESHOLD"])
        if "AUDIT_RATE" in os.environ:
            cfg.audit_rate = float(os.environ["AUDIT_RATE"])
        if "RAG_TOP_K" in os.environ:
            cfg.top_k = int(os.environ["RAG_TOP_K"])
        if "POLICY_PATH" in os.environ:
            cfg.policy_path = os.environ["POLICY_PATH"]
        if "CHROMA_DIR" in os.environ:
            cfg.chroma_dir = os.environ["CHROMA_DIR"]
        if "ADJUDICATION_STRATEGY" in os.environ:
            cfg.adjudication_strategy = os.environ["ADJUDICATION_STRATEGY"]
        if "SELF_CONSISTENCY_SAMPLES" in os.environ:
            cfg.self_consistency_samples = int(os.environ["SELF_CONSISTENCY_SAMPLES"])
        if "HARDEN_ADJUDICATOR" in os.environ:
            cfg.harden_adjudicator = os.environ["HARDEN_ADJUDICATOR"].lower() in ("1", "true", "yes", "on")
        if "NORMALIZE_TIER1" in os.environ:
            cfg.normalize_tier1 = os.environ["NORMALIZE_TIER1"].lower() in ("1", "true", "yes", "on")

        if not 0.0 <= cfg.audit_rate <= 1.0:
            raise ValueError(f"audit_rate must be within [0, 1], got {cfg.audit_rate}")
        if cfg.adjudication_strategy not in ("direct", "cot", "self_consistency"):
            raise ValueError(f"unknown adjudication_strategy: {cfg.adjudication_strategy!r}")
        return cfg
