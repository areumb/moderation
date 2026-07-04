"""Tier-1 classifier wrapper.

Two implementations behind one interface:

- StubClassifier: deterministic, dependency-free. Used when the MODEL_DIR
  environment variable is unset, so the whole service (and CI) runs offline
  with no trained weights.
- HFClassifier: loads the real fine-tuned model. It reuses the exact loading
  convention of the research code (hs_generalization): checkpoints saved by
  hs_generalization.utils.save_model are dicts with a "model" state-dict key,
  and the head is a RoBERTa AutoModelForSequenceClassification with 3 labels
  (base encoding 0=hate, 1=offensive, 2=clean — see hs_generalization/modes.py).

MODEL_DIR may point to:
  * a directory containing a .pt checkpoint produced by the thesis code
    (e.g. outputs/davidson/RoBERTa-base/3class/), optionally with MODEL_NAME
    naming the HF base model (default: "roberta-base"); or
  * a HuggingFace model directory (contains config.json) saved with
    save_pretrained().

Optionally, probabilities can be projected into the thesis' binary label
spaces (hate_nonhate, nonclean_clean) by summing class probabilities — the
same probability-merging rule as hs_generalization.modes.projection_for.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from serving.config import LABELS

logger = logging.getLogger(__name__)

# Probability-merge groups per binary view, mirroring
# hs_generalization.modes.projection_for (3class -> binary projections).
PROJECTIONS = {
    "hate_nonhate": {"groups": [[0], [1, 2]], "labels": ["Hateful", "Non-hateful"]},
    "nonclean_clean": {"groups": [[0, 1], [2]], "labels": ["Non-clean", "Clean"]},
}


class Classifier(Protocol):
    def predict(self, text: str) -> dict: ...


def _result(probs: list[float]) -> dict:
    top = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "label": LABELS[top],
        "probs": {LABELS[i]: round(float(probs[i]), 6) for i in range(len(probs))},
        "confidence": round(float(probs[top]), 6),
    }


def project(probs: dict[str, float], mode: str) -> dict:
    """Project ternary probabilities into a binary view by summing class probs.

    Same merging rule as hs_generalization.modes.projection_for, applied to
    the probability vector instead of logits.
    """
    if mode not in PROJECTIONS:
        raise ValueError(f"Unknown projection mode: {mode}")
    vec = [probs[label] for label in LABELS]
    spec = PROJECTIONS[mode]
    merged = [sum(vec[i] for i in group) for group in spec["groups"]]
    top = max(range(len(merged)), key=lambda i: merged[i])
    return {
        "mode": mode,
        "label": spec["labels"][top],
        "probs": {spec["labels"][i]: round(merged[i], 6) for i in range(len(merged))},
    }


class StubClassifier:
    """Deterministic stand-in for the fine-tuned model.

    Simple keyword heuristics over masked placeholders so tests and CI can
    exercise every routing path offline. It is NOT a hate speech detector and
    produces no meaningful predictions on real text.
    """

    name = "stub"

    # Masked/mild trigger tokens only — no real slurs (see project policy).
    _hateful_markers = ("[slur]", "[hateful]", "i hate all")
    _offensive_markers = ("[insult]", "[offensive]", "idiot", "trash take")
    _ambiguous_markers = ("[ambiguous]",)

    def predict(self, text: str) -> dict:
        t = text.lower()
        if any(m in t for m in self._ambiguous_markers):
            # Near-tie between Offensive and Hateful -> margin escalation.
            return _result([0.44, 0.46, 0.10])
        if any(m in t for m in self._hateful_markers):
            return _result([0.90, 0.07, 0.03])
        if any(m in t for m in self._offensive_markers):
            return _result([0.08, 0.82, 0.10])
        if "[uncertain]" in t:
            # Low-confidence Clean -> confidence escalation.
            return _result([0.20, 0.25, 0.55])
        return _result([0.02, 0.05, 0.93])


class HFClassifier:
    """Loads the real fine-tuned RoBERTa by reusing the research repo's
    checkpoint convention (see module docstring). Torch/transformers are
    imported lazily so the stub path has no heavy dependencies."""

    name = "hf"

    def __init__(self, model_dir: str, model_name: str | None = None, max_length: int = 512):
        import torch  # noqa: F401 — lazy heavy imports
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        model_dir_p = Path(model_dir)
        model_name = model_name or os.environ.get("MODEL_NAME", "roberta-base")
        self.max_length = max_length

        if (model_dir_p / "config.json").exists():
            # HF save_pretrained() directory.
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir_p)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir_p)
        else:
            # Thesis checkpoint: {"model": state_dict, ...} saved by
            # hs_generalization.utils.save_model; same loading steps as
            # hs_generalization/test.py.
            ckpts = sorted(model_dir_p.glob("*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No .pt checkpoint or config.json found in {model_dir}")
            ckpt_path = ckpts[0]
            logger.info("Loading thesis checkpoint %s (base model %s)", ckpt_path, model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.eval()

    def predict(self, text: str) -> dict:
        # Research code lowercases inputs at tokenization time
        # (hs_generalization.utils.get_dataset) — match that here.
        enc = self.tokenizer(text.lower(), truncation=True, max_length=self.max_length, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**enc).logits
        probs = logits.softmax(dim=-1)[0].tolist()
        return _result(probs)


def get_classifier() -> Classifier:
    """Factory: real model when MODEL_DIR is set, deterministic stub otherwise."""
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        return HFClassifier(model_dir)
    logger.warning("MODEL_DIR not set — using StubClassifier (deterministic, not a real model).")
    return StubClassifier()
