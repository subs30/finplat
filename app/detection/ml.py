import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.detection.base import Finding
from app.detection.features import FEATURE_COLUMNS

# Roles from scripts/seed_fraud_data.py's ground truth that represent an
# account actually exhibiting the suspicious activity (as opposed to an
# account that is merely *involved*, like a mule network's victim_source,
# or a deliberate clean/negative control). Only these roles get label=1;
# every other account in the org (including every unlabeled background
# account) is label=0. See the V0.3 Step 2 report for why victims aren't
# expected to be flagged themselves.
_POSITIVE_ROLES = {
    "structurer",
    "mule",
    "destination",
    "origin",
    "intermediary_1",
    "intermediary_2",
    "intermediary_3",
}

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "app" / "data" / "models" / "detection_model.json"


def load_labels(ground_truth_path: Path) -> dict[str, int]:
    """account_id (str) -> 1 if its ground-truth role means it should be
    flagged, 0 if it's explicitly a victim/clean-control role. Accounts
    not mentioned at all (the background population) get no entry here —
    callers should default those to 0, which build_training_table does.
    """
    gt = json.loads(ground_truth_path.read_text())
    labels: dict[str, int] = {}
    for pattern in gt["patterns"]:
        for account in pattern["accounts"]:
            label = 1 if account["role"] in _POSITIVE_ROLES else 0
            labels[account["account_id"]] = label
    return labels


def build_training_table(features: pd.DataFrame, labels: dict[str, int]) -> tuple[pd.DataFrame, pd.Series]:
    """Joins a feature table (from app.detection.features.build_feature_table)
    with ground-truth labels. Any account_id in `features` not present in
    `labels` (i.e. the unlabeled background population) is labeled 0.
    """
    y = pd.Series(
        [labels.get(account_id, 0) for account_id in features.index],
        index=features.index,
        name="label",
    )
    return features, y


def train_model(
    features: pd.DataFrame, labels: pd.Series, *, test_size: float = 0.3, random_state: int = 42
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    """Fits an XGBoost gradient boosting classifier on `features`/`labels`
    and evaluates it on a held-out stratified split. Returns the fitted
    model and an honest metrics dict — see app/detection/ml.py's module
    docstring in scripts/train_model.py for why these numbers should not
    be read as production performance figures.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    scale_pos_weight = (negative_count / positive_count) if positive_count else 1.0

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "train_size": len(x_train),
        "test_size": len(x_test),
        "train_positive_count": positive_count,
        "test_positive_count": int(y_test.sum()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "feature_importances": dict(
            zip(FEATURE_COLUMNS, [float(v) for v in model.feature_importances_], strict=True)
        ),
    }
    return model, metrics


def save_model(model: xgb.XGBClassifier, path: Path = _DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path = _DEFAULT_MODEL_PATH) -> xgb.XGBClassifier:
    """Raises FileNotFoundError (not xgboost's own error type) if `path`
    doesn't exist, so callers — notably the combined detection endpoint —
    can catch one predictable, standard exception to detect "model not
    trained yet" rather than depending on an xgboost internal.
    """
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path} — run scripts/train_model.py first.")
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def predict(model: xgb.XGBClassifier, features: pd.DataFrame, *, threshold: float = 0.5) -> list[Finding]:
    """Scores every account in `features` and returns a Finding for each
    one the model flags (probability >= threshold) — accounts it doesn't
    flag simply produce no Finding, same convention as the rules engine.
    """
    if features.empty:
        return []
    probabilities = model.predict_proba(features)[:, 1]
    findings = []
    for account_id, probability in zip(features.index, probabilities, strict=True):
        if probability >= threshold:
            findings.append(
                Finding(
                    method="ml:gradient_boosting",
                    account_id=uuid.UUID(account_id),
                    triggered=True,
                    explanation=(
                        f"Gradient boosting classifier scored this account {probability:.2f} "
                        f"(threshold {threshold}) based on transaction velocity, amount "
                        "patterns, and counterparty diversity."
                    ),
                    score=float(probability),
                )
            )
    return findings
