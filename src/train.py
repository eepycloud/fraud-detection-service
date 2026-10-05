"""Train the baseline LightGBM fraud model and report metrics on the test file."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import FEATURES, TARGET, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load(name):
    df = pd.read_csv(DATA / name).sort_values("trans_date_trans_time")
    return build_features(df), df[TARGET]


def main():
    X, y = load("fraudTrain.csv")
    X_test, y_test = load("fraudTest.csv")

    # Time-based split: the most recent 20% of the training period is used
    # for early stopping and threshold selection. The test file is untouched
    # until the final evaluation.
    cut = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_val, y_val = X.iloc[cut:], y.iloc[cut:]

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        metric="average_precision",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    # Pick the decision threshold that maximises F1 on the validation period.
    val_scores = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, val_scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    threshold = float(t[f1[:-1].argmax()])

    # Final evaluation on the held-out test file.
    scores = model.predict_proba(X_test)[:, 1]
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "trees": int(model.best_iteration_),
        "threshold": round(threshold, 4),
        "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "test_rows": int(len(y_test)),
        "test_fraud_rate": round(float(y_test.mean()), 5),
    }

    print("\nTest set results")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nFeature importance (gain)")
    gain = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=FEATURES
    )
    print((gain / gain.sum()).sort_values(ascending=False).round(3).to_string())

    MODELS.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": threshold, "features": FEATURES},
        MODELS / "baseline.joblib",
    )
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model and metrics to {MODELS}")


if __name__ == "__main__":
    main()
