# Trained Detection Model

`detection_model.json` is the XGBoost gradient boosting classifier from
V0.3's fraud/AML detection pipeline (`app/detection/ml.py`), trained on
the synthetic dataset from `scripts/seed_fraud_data.py` and the ground
truth in `../synthetic_fraud_data/ground_truth.json`.

**Committed, not gitignored** — it's ~55KB (XGBoost's native JSON format,
100 trees at max_depth=3 over 16 features), small enough that committing
it costs nothing and means the combined detection endpoint
(`GET /detection/accounts/{id}`) works immediately after cloning and
setting up the database, without a mandatory training step first.

**This is a demonstration artifact, not a production model.** It was
trained on one small synthetic organization (184 accounts, 17 positive
examples) — see the V0.3 report's Step 5/8 sections for the honest
precision/recall numbers and why they don't generalize.

To retrain (e.g. against a different seeded organization, or after
changing feature engineering):

```bash
python scripts/train_model.py --org-id <organization-uuid>
```

This overwrites `detection_model.json` in place.
