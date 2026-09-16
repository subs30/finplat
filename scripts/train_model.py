#!/usr/bin/env python3
"""Train the gradient boosting (XGBoost) fraud/AML detection model on one
organization's synthetic data and ground truth, and save it to
app/data/models/detection_model.json.

This is a demonstration pipeline, not a production model-training setup:
the synthetic dataset is small (~184 accounts, ~17 positive examples), so
the held-out precision/recall/F1 this script reports are illustrative of
the pipeline working end-to-end, not evidence of real-world detection
performance. See the V0.3 report's Step 5 section for the honest read on
these numbers.

Usage:
    python scripts/train_model.py --org-id <organization-uuid> --ground-truth app/data/synthetic_fraud_data/ground_truth.json
"""
import argparse
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.database import SessionLocal
from app.detection.features import build_feature_table
from app.detection.ml import build_training_table, load_labels, save_model, train_model

_DEFAULT_GROUND_TRUTH = (
    Path(__file__).resolve().parents[1] / "app" / "data" / "synthetic_fraud_data" / "ground_truth.json"
)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "models" / "detection_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True, type=uuid.UUID)
    parser.add_argument("--ground-truth", type=Path, default=_DEFAULT_GROUND_TRUTH)
    parser.add_argument("--model-out", type=Path, default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        features = build_feature_table(db, args.org_id, as_of=datetime.now(UTC))
    finally:
        db.close()

    labels_by_account = load_labels(args.ground_truth)
    features, labels = build_training_table(features, labels_by_account)

    print(f"Training table: {len(features)} accounts, {int(labels.sum())} positive")
    model, metrics = train_model(features, labels, test_size=args.test_size)

    print("\nHeld-out evaluation (small synthetic dataset — see script docstring):")
    print(f"  train: {metrics['train_size']} accounts ({metrics['train_positive_count']} positive)")
    print(f"  test:  {metrics['test_size']} accounts ({metrics['test_positive_count']} positive)")
    print(f"  precision: {metrics['precision']:.3f}")
    print(f"  recall:    {metrics['recall']:.3f}")
    print(f"  f1:        {metrics['f1']:.3f}")
    print("\nTop feature importances:")
    for name, importance in sorted(metrics["feature_importances"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {name:<30} {importance:.3f}")

    save_model(model, args.model_out)
    print(f"\nModel saved to {args.model_out}")


if __name__ == "__main__":
    main()
