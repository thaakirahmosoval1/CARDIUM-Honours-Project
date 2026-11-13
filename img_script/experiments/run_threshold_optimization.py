import sys
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.transformations import transform_train, transform_test
from data.load_img_data import create_image_dataloaders
from data.dataloader import CardiumDataset
from img_script.img_models.get_img_model import ImageModel
from utils import inference_image_model
from img_script.experiments.threshold_utils import (
    calculate_metrics,
    optimise_threshold,
)

CHECKPOINT_DIR = ROOT / "img_script/image_checkpoints/image_encoder_BASELINE"
OUTPUT_DIR = ROOT / "img_script/experiments/threshold_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

args = Namespace(
    img_model="vit_small",
    img_pretrain=False,
    n_classes=1,
    img_path_dropout=0.3,
    img_class_dropout=0.2,
    sampling=False,
)

all_results = []
all_thresholds = []

print("Device:", device)

for fold in range(3):
    print(f"\n{'=' * 55}")
    print(f"FOLD {fold + 1}/3")
    print(f"{'=' * 55}")

    dataset_dir = ROOT / f"dataset/cardium_images/fold_{fold + 1}"

    _, val_loader, test_loader = create_image_dataloaders(
        dataset_dir=str(dataset_dir),
        dataset_class=CardiumDataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=8,
        args=args,
    )

    model = ImageModel(args).build_model()

    checkpoint_path = CHECKPOINT_DIR / f"fold{fold}_best_model.pth"

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    print("\nGenerating validation predictions...")
    val_true, val_score = inference_image_model(
        model, val_loader, device
    )

    best_threshold, threshold_results = optimise_threshold(
        val_true, val_score
    )

    threshold_results.insert(0, "fold", fold + 1)
    all_thresholds.append(threshold_results)

    print(f"\nSelected validation threshold: {best_threshold:.2f}")

    print("\nGenerating held-out test predictions...")
    test_true, test_score = inference_image_model(
        model, test_loader, device
    )

    for threshold_type, threshold in [
        ("default_0.5", 0.5),
        ("optimised", best_threshold),
    ]:
        metrics = calculate_metrics(
            test_true, test_score, threshold
        )

        metrics["fold"] = fold + 1
        metrics["method"] = threshold_type
        metrics["validation_selected_threshold"] = best_threshold

        all_results.append(metrics)

        print(
            f"{threshold_type}: "
            f"F1={metrics['f1']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Specificity={metrics['specificity']:.4f}"
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

threshold_df = pd.concat(all_thresholds, ignore_index=True)
comparison_df = pd.DataFrame(all_results)

summary_df = (
    comparison_df
    .groupby("method")[
        ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    ]
    .agg(["mean", "std"])
)

threshold_df.to_csv(
    OUTPUT_DIR / "threshold_search.csv", index=False
)

comparison_df.to_csv(
    OUTPUT_DIR / "fold_comparison.csv", index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "summary.csv"
)

print("\nFINAL SUMMARY")
print(summary_df.to_string())

print(f"\nResults saved to: {OUTPUT_DIR}")
