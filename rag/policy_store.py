"""Policy vector store.

Parses policies/community_guidelines.md into clauses with stable ids
(### <ID>: <title> headings), embeds them, and persists them in a local
Chroma collection. Building is idempotent: the collection stores a hash of
the policy file and is rebuilt only when the file changes. Swapping the
policy document therefore changes the system's decisions with no retraining —
the point of making retrieval load-bearing.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import get_embedder

logger = logging.getLogger(__name__)

CLAUSE_RE = re.compile(r"^###\s+(?P<id>[A-Z]{2,4}-\d+):\s*(?P<title>.+)$")


@dataclass
class Clause:
    clause_id: str
    title: str
    text: str
    category: str  # Hateful | Offensive | Clean | Definition

    @property
    def document(self) -> str:
        return f"{self.clause_id}: {self.title}\n{self.text}"


_PREFIX_TO_CATEGORY = {"HL": "Hateful", "OF": "Offensive", "CL": "Clean", "DEF": "Definition"}


def parse_policy(path: str | Path) -> list[Clause]:
    """Split the policy markdown into clauses keyed by their stable ids."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    clauses: list[Clause] = []
    current: dict | None = None

    def flush():
        if current is not None:
            body = "\n".join(current["body"]).strip()
            prefix = current["id"].split("-")[0]
            clauses.append(
                Clause(
                    clause_id=current["id"],
                    title=current["title"],
                    text=body,
                    category=_PREFIX_TO_CATEGORY.get(prefix, "Other"),
                )
            )

    for line in lines:
        m = CLAUSE_RE.match(line)
        if m:
            flush()
            current = {"id": m.group("id"), "title": m.group("title").strip(), "body": []}
        elif line.startswith("## ") or line.startswith("# "):
            flush()
            current = None
        elif current is not None:
            current["body"].append(line)
    flush()

    if not clauses:
        raise ValueError(f"No clauses found in policy file {path}")
    return clauses


class PolicyStore:
    def __init__(self, policy_path: str | Path, chroma_dir: str | Path, collection_name: str):
        import chromadb

        self.policy_path = Path(policy_path)
        self.clauses = parse_policy(self.policy_path)
        self.by_id = {c.clause_id: c for c in self.clauses}
        self.embedder = get_embedder()

        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection_name = collection_name
        self.collection = self._build_if_needed()

    def _policy_hash(self) -> str:
        payload = self.policy_path.read_bytes() + self.embedder.backend_name.encode()
        return hashlib.sha256(payload).hexdigest()

    def _build_if_needed(self):
        policy_hash = self._policy_hash()
        try:
            col = self._client.get_collection(self.collection_name, embedding_function=self.embedder)
            if col.metadata and col.metadata.get("policy_hash") == policy_hash:
                logger.info("Policy index up to date (%d clauses).", col.count())
                return col
            logger.info("Policy file or embedder changed — rebuilding index.")
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection does not exist yet

        col = self._client.create_collection(
            self.collection_name,
            embedding_function=self.embedder,
            metadata={"policy_hash": policy_hash, "embedder": self.embedder.backend_name},
        )
        col.add(
            ids=[c.clause_id for c in self.clauses],
            documents=[c.document for c in self.clauses],
            metadatas=[{"category": c.category, "title": c.title} for c in self.clauses],
        )
        logger.info("Indexed %d policy clauses from %s.", len(self.clauses), self.policy_path)
        return col

    def query(self, text: str, top_k: int) -> list[dict]:
        res = self.collection.query(query_texts=[text], n_results=min(top_k, len(self.clauses)))
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                {
                    "clause_id": cid,
                    "text": doc,
                    "category": meta.get("category"),
                    "title": meta.get("title"),
                    "distance": float(dist),
                }
            )
        return out
