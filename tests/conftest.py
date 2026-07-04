"""Shared fixtures. Everything runs offline: StubClassifier + MockLLM,
Chroma in a temp dir, hashed-BoW embedding fallback if sentence-transformers
is not installed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def offline_env(tmp_path_factory):
    """Guarantee the offline path regardless of the developer's environment."""
    import os

    for var in ("MODEL_DIR", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        os.environ.pop(var, None)
    os.environ["CHROMA_DIR"] = str(tmp_path_factory.mktemp("chroma"))
    # Audit sampling off for the shared client so route assertions are
    # deterministic; the audit path has its own dedicated client/tests.
    os.environ["AUDIT_RATE"] = "0.0"
    return os.environ


@pytest.fixture(scope="session")
def client(offline_env):
    from fastapi.testclient import TestClient

    from serving.app import app

    with TestClient(app) as c:  # context manager triggers lifespan
        yield c


@pytest.fixture(scope="session")
def policy_store(offline_env, tmp_path_factory):
    from rag.policy_store import PolicyStore

    return PolicyStore(
        REPO_ROOT / "policies" / "community_guidelines.md",
        tmp_path_factory.mktemp("chroma-store"),
        "test_guidelines",
    )
