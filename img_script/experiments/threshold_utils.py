import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def calculate_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_score = np.asarray(y_score).reshape(-1)

    y_pred = (y_score > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": (
            roc_auc_score(y_true, y_score)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def optimise_threshold(y_true, y_score):
    thresholds = np.round(
        np.arange(0.20, 0.801, 0.01), 2
    )

    results = pd.DataFrame([
        calculate_metrics(y_true, y_score, threshold)
        for threshold in thresholds
    ])

    # Highest validation F1, then threshold closest to 0.5
    results["distance_from_default"] = (
        results["threshold"] - 0.5
    ).abs()

    ranked = results.sort_values(
        ["f1", "distance_from_default", "threshold"],
        ascending=[False, True, True]
    )

    best_threshold = float(ranked.iloc[0]["threshold"])

    return best_threshold, results.drop(
        columns="distance_from_default"
    )
