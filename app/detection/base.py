import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Finding:
    """One detection method's verdict on one account — the shared result
    shape across rules, ML, and graph detection (see app/detection/rules.py,
    ml.py, graph.py), so app/routers/detection.py can combine them without
    each method needing to know about the others.

    Explainability is load-bearing here, same as the RAG pipeline's
    citations: `explanation` and `evidence_transaction_ids` must let a
    human verify the finding against the actual data, not just trust a
    label.
    """

    method: str  # e.g. "rule:structuring", "ml:gradient_boosting", "graph:layering_chain"
    account_id: uuid.UUID
    triggered: bool
    explanation: str
    evidence_transaction_ids: list[uuid.UUID] = field(default_factory=list)
    score: float | None = None  # ML confidence, or None for rules/graph (binary by nature)
    detected_at: datetime | None = None
