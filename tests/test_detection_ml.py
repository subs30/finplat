import json
import uuid

import pandas as pd
import pytest

from app.detection.features import FEATURE_COLUMNS
from app.detection.ml import (
    build_training_table,
    load_labels,
    load_model,
    predict,
    save_model,
    train_model,
)


def test_load_labels_maps_flaggable_roles_to_1_and_others_to_0(tmp_path):
    ground_truth = {
        "patterns": [
            {
                "pattern_type": "structuring",
                "accounts": [{"account_id": "aaaa", "role": "structurer"}],
            },
            {
                "pattern_type": "mule_network",
                "accounts": [
                    {"account_id": "bbbb", "role": "mule"},
                    {"account_id": "cccc", "role": "victim_source"},
                ],
            },
            {
                "pattern_type": "clean",
                "accounts": [{"account_id": "dddd", "role": "clean_control"}],
            },
        ]
    }
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(ground_truth))

    labels = load_labels(path)

    assert labels == {"aaaa": 1, "bbbb": 1, "cccc": 0, "dddd": 0}


def _synthetic_feature_table(n_positive: int = 8, n_negative: int = 40) -> tuple[pd.DataFrame, dict[str, int]]:
    """A small, cheap-to-train fixture with an obviously separable signal
    (positive rows have much higher txn_count/near_threshold_deposit_count)
    — not the real 184-account synthetic dataset. Training on this in CI
    takes a fraction of a second and tests the pipeline's mechanics
    (label join, split, fit, predict, save/load), not detection quality —
    see scripts/train_model.py for the real dataset's honest performance
    numbers.
    """
    rows: dict[str, dict[str, float]] = {}
    labels: dict[str, int] = {}

    for i in range(n_positive):
        account_id = str(uuid.uuid4())
        rows[account_id] = {col: 1.0 for col in FEATURE_COLUMNS}
        rows[account_id].update({"txn_count": 20.0, "near_threshold_deposit_count": 5.0, "total_amount": 40000.0})
        labels[account_id] = 1

    for i in range(n_negative):
        account_id = str(uuid.uuid4())
        rows[account_id] = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        rows[account_id].update({"txn_count": 2.0, "total_amount": 200.0})
        labels[account_id] = 0

    df = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_COLUMNS)
    df.index.name = "account_id"
    return df, labels


def test_build_training_table_defaults_unlabeled_accounts_to_zero():
    features, labels = _synthetic_feature_table(n_positive=2, n_negative=3)
    # Drop one label to simulate an unlabeled background account.
    some_account = next(iter(labels))
    labels_missing_one = {k: v for k, v in labels.items() if k != some_account}

    _, y = build_training_table(features, labels_missing_one)

    assert y[some_account] == 0
    assert y.sum() == sum(labels.values()) - (1 if labels[some_account] == 1 else 0)


def test_train_predict_save_load_roundtrip(tmp_path):
    features, labels = _synthetic_feature_table()
    features, y = build_training_table(features, labels)

    model, metrics = train_model(features, y, test_size=0.3, random_state=1)

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert set(metrics["feature_importances"]) == set(FEATURE_COLUMNS)

    # The synthetic signal is obviously separable, so the model should
    # recover it well on this fixture — this is a pipeline sanity check,
    # not a claim about real-world detection quality.
    assert metrics["recall"] > 0.5

    model_path = tmp_path / "model.json"
    save_model(model, model_path)
    assert model_path.exists()

    loaded = load_model(model_path)
    findings = predict(loaded, features, threshold=0.5)
    flagged_ids = {str(f.account_id) for f in findings}
    true_positive_ids = {aid for aid, label in labels.items() if label == 1}
    # Every obviously-positive synthetic row should be recovered.
    assert true_positive_ids <= flagged_ids


def test_load_model_raises_file_not_found_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does-not-exist.json")
