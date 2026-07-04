"""Top-k clause retrieval over the policy store."""
from __future__ import annotations

from rag.policy_store import PolicyStore


class Retriever:
    def __init__(self, store: PolicyStore, top_k: int = 3):
        self.store = store
        self.top_k = top_k

    def retrieve(self, text: str, classifier_hint: str | None = None) -> list[dict]:
        """Retrieve the clauses most relevant to `text`.

        If the classifier's tentative label is provided, the query is expanded
        with the two candidate categories at the hard boundary (Offensive vs
        Hateful) so the adjudicator always sees both sides of the distinction
        it has to make.
        """
        query = text
        if classifier_hint in ("Hateful", "Offensive"):
            query = (
                f"{text}\n(candidate categories: hateful group-based attack "
                "vs. non-group-based offensive insult)"
            )
        return self.store.query(query, self.top_k)
