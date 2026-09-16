import uuid
from collections import defaultdict

from neo4j import Driver

from app.detection.base import Finding

# --- layering: variable-length path traversal -----------------------------

LAYERING_MIN_HOPS = 3
LAYERING_MAX_HOPS = 6
LAYERING_MIN_RETENTION = 0.5
LAYERING_MAX_GROWTH = 1.05
LAYERING_MAX_HOP_GAP_HOURS = 96
# Real layering moves conspicuously large sums relative to ordinary
# transaction activity — without a floor here, a long enough random walk
# through background noise will occasionally satisfy the retention-ratio
# and time-gap constraints by chance on perfectly ordinary small transfers
# (observed directly against this dataset: dropping this filter produced
# several $900-$2,900 "chains" from background noise alongside the two
# real $44k-$60k embedded chains). $15,000 sits comfortably above
# background's occasional larger-but-legitimate transactions (up to
# $8,000) and comfortably below the embedded chains' starting amounts.
LAYERING_MIN_FIRST_HOP_AMOUNT = 15000

_LAYERING_QUERY = """
MATCH path = (origin:Account {organization_id: $org_id})-[rels:TRANSACTED*%(min_hops)d..%(max_hops)d]->(dest:Account {organization_id: $org_id})
WHERE origin <> dest
  AND all(n IN nodes(path) WHERE n.organization_id = $org_id)
  AND rels[0].amount >= $min_first_hop_amount
  AND all(idx IN range(0, size(rels)-2) WHERE rels[idx].occurred_at <= rels[idx+1].occurred_at)
  AND all(idx IN range(0, size(rels)-2) WHERE
        rels[idx+1].amount <= rels[idx].amount * $max_growth
        AND rels[idx+1].amount >= rels[idx].amount * $min_retention)
  AND all(idx IN range(0, size(rels)-2) WHERE
        duration.between(rels[idx].occurred_at, rels[idx+1].occurred_at).hours <= $max_hop_gap_hours)
RETURN [n IN nodes(path) | n.id] AS account_ids,
       [r IN rels | r.transaction_id] AS transaction_ids,
       [r IN rels | r.amount] AS amounts
ORDER BY size(rels) DESC
LIMIT 200
"""


def detect_layering_chains(
    driver: Driver,
    organization_id: uuid.UUID,
    *,
    min_hops: int = LAYERING_MIN_HOPS,
    max_hops: int = LAYERING_MAX_HOPS,
    min_retention: float = LAYERING_MIN_RETENTION,
    max_growth: float = LAYERING_MAX_GROWTH,
    max_hop_gap_hours: int = LAYERING_MAX_HOP_GAP_HOURS,
    min_first_hop_amount: float = LAYERING_MIN_FIRST_HOP_AMOUNT,
    database: str = "neo4j",
) -> list[Finding]:
    """Finds chains of `min_hops`-`max_hops` TRANSACTED edges, starting at
    or above `min_first_hop_amount`, where the amount is roughly preserved
    hop-to-hop (between `min_retention` and `max_growth` of the previous
    hop) and each hop happens at or after the previous one, within
    `max_hop_gap_hours` of it — layering's rapid-pass-through-with-
    minimal-loss signature.

    This genuinely cannot be done at the single-account scope the Step 4
    rules operate at: it requires tracing an arbitrary-length path across
    the whole account graph, which is exactly what a graph database's
    variable-length pattern matching is for and what a relational query
    would need recursive CTEs (and still no natural notion of "path") to
    approximate.

    Two things Cypher's variable-length matching does NOT give you for
    free, both handled after the query returns:

    1. Node uniqueness. `-[:TYPE*n..m]->` only guarantees each
       *relationship* is used once per path, not each *node* — a path can
       legitimately revisit an account via a different edge. A layering
       chain revisiting an account isn't layering (that's a cycle, a
       different pattern), so `_dedupe_to_longest_chains` drops any
       returned path with a repeated account.
    2. Sub-path collapsing. Variable-length matching returns every path
       length in range, so a genuine 6-hop chain also produces
       valid-looking 3, 4, and 5-hop sub-paths starting partway through
       it — `_dedupe_to_longest_chains` also collapses those down to the
       longest chain per connected node-set so one real chain produces
       one Finding, not four overlapping ones.
    """
    query = _LAYERING_QUERY % {"min_hops": min_hops, "max_hops": max_hops}
    with driver.session(database=database) as session:
        result = session.run(
            query,
            org_id=str(organization_id),
            min_retention=min_retention,
            max_growth=max_growth,
            max_hop_gap_hours=max_hop_gap_hours,
            min_first_hop_amount=min_first_hop_amount,
        )
        raw_chains = [dict(record) for record in result]

    raw_chains = [c for c in raw_chains if len(set(c["account_ids"])) == len(c["account_ids"])]
    chains = _dedupe_to_longest_chains(raw_chains)

    findings = []
    for chain in chains:
        account_ids = chain["account_ids"]
        amounts = chain["amounts"]
        retained_pct = (amounts[-1] / amounts[0] * 100) if amounts[0] else 0.0
        # One Finding per account in the chain — each hop is itself part of
        # the suspicious pattern, not just the endpoints (see the V0.3 data
        # model report: origin, every intermediary, and the destination are
        # all ground-truth-flaggable roles).
        for account_id in account_ids:
            findings.append(
                Finding(
                    method="graph:layering_chain",
                    account_id=uuid.UUID(account_id),
                    triggered=True,
                    explanation=(
                        f"Part of a {len(account_ids)}-account, {len(amounts)}-hop transfer "
                        f"chain retaining {retained_pct:.0f}% of the original "
                        f"${amounts[0]:,.2f} by the final hop, each hop within "
                        f"{max_hop_gap_hours}h of the previous one — a layering pattern."
                    ),
                    evidence_transaction_ids=[uuid.UUID(t) for t in chain["transaction_ids"]],
                )
            )
    return findings


def _dedupe_to_longest_chains(raw_chains: list[dict]) -> list[dict]:
    """Keeps only chains whose full account set isn't already covered by a
    longer chain already kept — see detect_layering_chains's docstring.
    """
    raw_chains = sorted(raw_chains, key=lambda c: len(c["account_ids"]), reverse=True)
    kept: list[dict] = []
    covered_node_sets: list[set] = []
    for chain in raw_chains:
        node_set = set(chain["account_ids"])
        if any(node_set <= covered for covered in covered_node_sets):
            continue
        kept.append(chain)
        covered_node_sets.append(node_set)
    return kept


# --- mule community: weakly-connected-component + fan-in/fan-out ----------

COMMUNITY_MIN_SIZE = 3
COMMUNITY_MAX_SIZE = 20
COMMUNITY_MIN_HUB_INDEGREE = 4

_FETCH_EDGES_QUERY = """
MATCH (sender:Account {organization_id: $org_id})-[t:TRANSACTED]->(receiver:Account {organization_id: $org_id})
RETURN sender.id AS sender_id, receiver.id AS receiver_id, t.transaction_id AS transaction_id
"""


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def detect_mule_communities(
    driver: Driver,
    organization_id: uuid.UUID,
    *,
    min_size: int = COMMUNITY_MIN_SIZE,
    max_size: int = COMMUNITY_MAX_SIZE,
    min_hub_indegree: int = COMMUNITY_MIN_HUB_INDEGREE,
    database: str = "neo4j",
) -> list[Finding]:
    """Finds weakly-connected components of the internal transfer graph
    sized between `min_size` and `max_size` accounts, then flags any
    account within such a component that receives from `min_hub_indegree`+
    other members of its OWN component — a small, structurally isolated
    cluster built around a fan-in hub, the shape of a mule network.

    Genuinely graph-native, distinct from the Step 4 fan-in/fan-out rule:
    that rule looks at one account's own in/out edges in isolation and
    would flag a hub sitting inside a 150-account, densely-interconnected
    background population exactly the same as one sitting in an isolated
    8-account cluster. This method's finding is about the SHAPE of the
    whole neighborhood the hub sits in — a small isolated component — which
    has no meaning at the single-account level a SQL query operates at.
    """
    with driver.session(database=database) as session:
        edges = [dict(record) for record in session.run(_FETCH_EDGES_QUERY, org_id=str(organization_id))]

    uf = _UnionFind()
    in_edges: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        uf.union(e["sender_id"], e["receiver_id"])
        in_edges[e["receiver_id"]].append(e)

    components: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        root = uf.find(e["sender_id"])
        components[root].add(e["sender_id"])
        components[root].add(e["receiver_id"])

    findings: list[Finding] = []
    for members in components.values():
        if not (min_size <= len(members) <= max_size):
            continue
        for account_id in members:
            in_component_senders = {
                e["sender_id"] for e in in_edges.get(account_id, []) if e["sender_id"] in members
            }
            if len(in_component_senders) >= min_hub_indegree:
                # Every internal edge of the component, not just the hub's
                # inbound ones — the fan-out leg (hub -> destination) is
                # just as much a part of the evidence as the fan-in.
                evidence_txn_ids = [
                    uuid.UUID(e["transaction_id"])
                    for e in edges
                    if e["sender_id"] in members and e["receiver_id"] in members
                ]
                for member_id in members:
                    findings.append(
                        Finding(
                            method="graph:mule_community",
                            account_id=uuid.UUID(member_id),
                            triggered=True,
                            explanation=(
                                f"Member of an isolated {len(members)}-account cluster "
                                f"(no connections to any other account) built around a hub "
                                f"receiving from {len(in_component_senders)} other members of "
                                "the same cluster — the fan-in shape of a mule network, "
                                "visible only by tracing the whole connected component, not "
                                "any single account's own transactions."
                            ),
                            evidence_transaction_ids=evidence_txn_ids,
                        )
                    )
                break  # one hub finding covers the whole component
    return findings


def run_all_graph_detection(driver: Driver, organization_id: uuid.UUID) -> list[Finding]:
    return detect_layering_chains(driver, organization_id) + detect_mule_communities(
        driver, organization_id
    )
