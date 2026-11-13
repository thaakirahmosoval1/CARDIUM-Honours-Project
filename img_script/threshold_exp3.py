import os
import sys
import csv
import pathlib
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.transformations import transform_train, transform_test
from data.load_img_data import create_image_dataloaders
from data.dataloader import CardiumDataset
from img_script.img_models.get_img_model import ImageModel
from utils import get_main_parser, set_seed, inference_image_model

CHECKPOINT_DIR = (
    ROOT / "img_script/image_checkpoints/2026-11-11-17-08-55"
)
OUTPUT_DIR = ROOT / "img_script/experiments/threshold_exp3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_metrics(y_true, y_score, threshold):
    predictions = (y_score >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )
    accuracy = accuracy_score(y_true, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def main():
    args = get_main_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    all_results = []
    all_predictions = []

    for fold in range(3):
        print(f"\nProcessing Fold {fold + 1}")
        set_seed(42)

        dataset_path = (
            f"{args.image_folder_path}/fold_{fold + 1}"
        )

        _, val_loader, _ = create_image_dataloaders(
            dataset_dir=dataset_path,
            dataset_class=CardiumDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            args=args,
        )

        model = ImageModel(args).build_model().to(device)

        checkpoint_path = (
            CHECKPOINT_DIR / f"fold{fold}_best_model.pth"
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        with torch.no_grad():
            y_true, y_score = inference_image_model(
                model, val_loader, device
            )

        y_true = np.asarray(y_true).reshape(-1).astype(int)
        y_score = np.asarray(y_score).reshape(-1)

        assert len(y_true) == len(y_score)
        assert np.all(np.isfinite(y_score))
        assert np.all((y_score >= 0) & (y_score <= 1))

        for patient_index, (label, score) in enumerate(
            zip(y_true, y_score)
        ):
            all_predictions.append({
                "fold": fold + 1,
                "patient_index": patient_index,
                "true_label": int(label),
                "probability": float(score),
            })

        for threshold in thresholds:
            metrics = calculate_metrics(
                y_true, y_score, threshold
            )
            all_results.append({
                "fold": fold + 1,
                "threshold": float(threshold),
                **metrics,
            })

    summary = []

    for threshold in thresholds:
        rows = [
            r for r in all_results
            if np.isclose(r["threshold"], threshold)
        ]

        summary.append({
            "threshold": float(threshold),
            "mean_f1": np.mean([r["f1"] for r in rows]),
            "mean_recall": np.mean(
                [r["recall"] for r in rows]
            ),
            "mean_precision": np.mean(
                [r["precision"] for r in rows]
            ),
            "mean_specificity": np.mean(
                [r["specificity"] for r in rows]
            ),
        })

    # Select highest mean F1, then recall as tie-breaker.
    best = max(
        summary,
        key=lambda r: (
            r["mean_f1"],
            r["mean_recall"]
        )
    )

    def save_csv(filename, rows):
        if not rows:
            return
        with open(
            OUTPUT_DIR / filename, "w", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    save_csv("fold_threshold_metrics.csv", all_results)
    save_csv("threshold_summary.csv", summary)
    save_csv("validation_predictions.csv", all_predictions)

    print("\nTHRESHOLD COMPARISON")
    print(
        f"{'Threshold':<12}"
        f"{'F1':<12}"
        f"{'Recall':<12}"
        f"{'Precision':<12}"
        f"{'Specificity':<12}"
    )

    for row in summary:
        print(
            f"{row['threshold']:<12.2f}"
            f"{row['mean_f1']:<12.4f}"
            f"{row['mean_recall']:<12.4f}"
            f"{row['mean_precision']:<12.4f}"
            f"{row['mean_specificity']:<12.4f}"
        )

    print("\nBEST VALIDATION THRESHOLD")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Mean F1: {best['mean_f1']:.4f}")
    print(f"Mean Recall: {best['mean_recall']:.4f}")
    print(f"Mean Precision: {best['mean_precision']:.4f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
