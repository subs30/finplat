#!/usr/bin/env python3
"""Run all three detection methods (rules, ML, graph) against the full
synthetic dataset for one organization and score them against the
ground-truth fixture — an honest per-method and per-pattern report, not
just an aggregate score. See the V0.3 report's Step 8 section for how to
read these numbers.

Usage:
    python scripts/verify_detection.py --org-id <organization-uuid>
"""
import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.database import SessionLocal
from app.detection.features import build_feature_table
from app.detection.graph import detect_layering_chains, detect_mule_communities
from app.detection.ml import load_model, predict
from app.detection.rules import detect_fanin_fanout, detect_structuring
from app.graph.dependency import get_neo4j_driver

_DEFAULT_GROUND_TRUTH = (
    Path(__file__).resolve().parents[1] / "app" / "data" / "synthetic_fraud_data" / "ground_truth.json"
)
_NOT_FLAGGABLE_ROLES = {"victim_source", "clean_control"}


def _load_ground_truth(path: Path) -> dict:
    return json.loads(path.read_text())


def _flaggable_and_not(gt: dict) -> tuple[set[str], set[str]]:
    flaggable: set[str] = set()
    not_flaggable: set[str] = set()
    for pattern in gt["patterns"]:
        for account in pattern["accounts"]:
            if account["role"] in _NOT_FLAGGABLE_ROLES:
                not_flaggable.add(account["account_id"])
            else:
                flaggable.add(account["account_id"])
    return flaggable, not_flaggable


def _score(
    name: str, flagged: set[str], flaggable: set[str], not_flaggable: set[str], all_accounts: set[str]
) -> dict:
    true_positive = flagged & flaggable
    false_positive_labeled = flagged & not_flaggable
    background = all_accounts - flaggable - not_flaggable
    false_positive_background = flagged & background
    false_negative = flaggable - flagged

    precision = len(true_positive) / len(flagged) if flagged else None
    recall = len(true_positive) / len(flaggable) if flaggable else None

    return {
        "method": name,
        "flagged_count": len(flagged),
        "true_positives": len(true_positive),
        "false_positives_background": len(false_positive_background),
        "false_positives_victim_or_clean": len(false_positive_labeled),
        "false_negatives": len(false_negative),
        "missed_accounts": sorted(false_negative),
        "precision": precision,
        "recall": recall,
    }


def _pattern_coverage(gt: dict, all_flagged_by_method: dict[str, set[str]]) -> list[dict]:
    rows = []
    for pattern in gt["patterns"]:
        flaggable_ids = {
            a["account_id"] for a in pattern["accounts"] if a["role"] not in _NOT_FLAGGABLE_ROLES
        }
        if not flaggable_ids:
            continue  # clean-control patterns have nothing flaggable to cover
        row = {"pattern_type": pattern["pattern_type"], "case_id": pattern["case_id"]}
        for method, flagged in all_flagged_by_method.items():
            caught = flaggable_ids & flagged
            row[method] = f"{len(caught)}/{len(flaggable_ids)}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True, type=uuid.UUID)
    parser.add_argument("--ground-truth", type=Path, default=_DEFAULT_GROUND_TRUTH)
    args = parser.parse_args()

    gt = _load_ground_truth(args.ground_truth)
    flaggable, not_flaggable = _flaggable_and_not(gt)

    db = SessionLocal()
    try:
        rule_findings = detect_structuring(db, args.org_id) + detect_fanin_fanout(db, args.org_id)
        rule_flagged = {str(f.account_id) for f in rule_findings}

        features = build_feature_table(db, args.org_id, as_of=datetime.now(UTC))
        all_accounts = set(features.index)
        try:
            model = load_model()
            ml_findings = predict(model, features)
            ml_flagged = {str(f.account_id) for f in ml_findings}
            ml_available = True
        except FileNotFoundError:
            ml_flagged = set()
            ml_available = False
    finally:
        db.close()

    try:
        driver = get_neo4j_driver()
        graph_findings = detect_layering_chains(driver, args.org_id) + detect_mule_communities(
            driver, args.org_id
        )
        graph_flagged = {str(f.account_id) for f in graph_findings}
        driver.close()
        graph_available = True
    except Exception:  # noqa: BLE001 - report unavailability, don't crash the report
        graph_flagged = set()
        graph_available = False

    combined_flagged = rule_flagged | ml_flagged | graph_flagged

    methods = {
        "rules": rule_flagged,
        "ml": ml_flagged,
        "graph": graph_flagged,
        "combined (any method)": combined_flagged,
    }

    print(f"Ground truth: {len(flaggable)} flaggable accounts, {len(not_flaggable)} victim/clean-control "
          f"(not expected to be flagged), {len(all_accounts) - len(flaggable) - len(not_flaggable)} "
          "unlabeled background accounts.\n")
    if not ml_available:
        print("NOTE: no trained model found — run scripts/train_model.py first. ML scored as 0 findings.\n")
    if not graph_available:
        print("NOTE: Neo4j unavailable — graph scored as 0 findings.\n")

    print(f"{'method':<24}{'flagged':>8}{'TP':>6}{'FP(bg)':>8}{'FP(vic/clean)':>15}{'FN':>6}{'precision':>11}{'recall':>9}")
    for name, flagged in methods.items():
        s = _score(name, flagged, flaggable, not_flaggable, all_accounts)
        precision_str = f"{s['precision']:.2f}" if s["precision"] is not None else "n/a"
        recall_str = f"{s['recall']:.2f}" if s["recall"] is not None else "n/a"
        print(
            f"{name:<24}{s['flagged_count']:>8}{s['true_positives']:>6}"
            f"{s['false_positives_background']:>8}{s['false_positives_victim_or_clean']:>15}"
            f"{s['false_negatives']:>6}{precision_str:>11}{recall_str:>9}"
        )
        if s["missed_accounts"]:
            print(f"    missed: {s['missed_accounts']}")

    print("\nPer-pattern coverage (flaggable accounts caught / total flaggable in that pattern):")
    coverage = _pattern_coverage(gt, methods)
    for row in coverage:
        print(
            f"  [{row['pattern_type']:<12}] {row['case_id']:<20} "
            f"rules={row['rules']:<6} ml={row['ml']:<6} graph={row['graph']:<6} "
            f"combined={row['combined (any method)']}"
        )


if __name__ == "__main__":
    main()
