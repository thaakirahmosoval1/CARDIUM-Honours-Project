import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
import sys
import os
import numpy as np

CARDIUM_path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(CARDIUM_path))

from multimodal_script.multimodal_models.get_multimodal_model import MultimodalModel
from data.load_multimodal_data import create_multimodal_dataloaders
from utils import *
from data.transformations import transform_train, transform_test
from img_script.img_models.get_img_model import ImageModel
from tabular_script.tab_models.get_tab_model import TabularModel
from multimodal_script.run_multimodal import MultimodalTrainer
from data.dataloader import CardiumDataset


args = get_main_parser()


def main(args):

    folds = args.folds

    test_metrics = {
        "F1 Score": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": []
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for fold in range(folds):

        print(f"\n====================================")
        print(f"Evaluating official checkpoint: Fold {fold + 1}/{folds}")
        print(f"====================================\n")

        # --------------------------------------------------
        # DATA
        # --------------------------------------------------

        dataset_path = f"{args.image_folder_path}/fold_{fold+1}"
        json_path = args.json_path

        set_seed(42)

        train_loader, test_loader = create_multimodal_dataloaders(
            dataset_dir=dataset_path,
            json_root=json_path,
            dataset_class=CardiumDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            args=args
        )

        print("Dataloaders created.")

        # --------------------------------------------------
        # IMAGE ENCODER
        # --------------------------------------------------

        img_model = ImageModel(args).build_model().to(device)

        image_checkpoint = os.path.join(
            args.img_checkpoint,
            f"fold{fold}_best_model.pth"
        )

        print(f"Loading image checkpoint: {image_checkpoint}")

        checkpoint = torch.load(
            image_checkpoint,
            map_location=device,
            weights_only=True
        )

        img_model.load_state_dict(checkpoint, strict=False)

        # Remove image classification head
        if args.img_model == "medvit":
            img_model.proj_head = nn.Identity()
        elif args.img_model in ["resnet18", "resnet50"]:
            img_model.fc = nn.Identity()
        else:
            img_model.head = nn.Identity()

        # --------------------------------------------------
        # TABULAR ENCODER
        # --------------------------------------------------

        tab_model = TabularModel(args).build_model().to(device)

        tab_checkpoint = os.path.join(
            args.tab_checkpoint,
            f"fold{fold}_best_model.pth"
        )

        print(f"Loading tabular checkpoint: {tab_checkpoint}")

        checkpoint = torch.load(
            tab_checkpoint,
            map_location=device,
            weights_only=True
        )

        tab_model.load_state_dict(checkpoint, strict=False)

        # Remove tabular classification head
        tab_model.mlp = nn.Identity()

        # --------------------------------------------------
        # MULTIMODAL MODEL
        # --------------------------------------------------

        multimodal_model = MultimodalModel(
            img_model=img_model,
            tab_model=tab_model,
            args=args
        ).build_model().to(device)

        multi_checkpoint = os.path.join(
            args.multimodal_checkpoint,
            f"fold{fold}_best_model.pth"
        )

        print(f"Loading OFFICIAL multimodal checkpoint: {multi_checkpoint}")

        checkpoint = torch.load(
            multi_checkpoint,
            map_location=device,
            weights_only=True
        )

        multimodal_model.load_state_dict(checkpoint, strict=False)

        # --------------------------------------------------
        # TRAINER
        # --------------------------------------------------

        loss_weights = torch.tensor(args.loss_factor).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=loss_weights
        ).to(device)

        # Optimizer is required by MultimodalTrainer,
        # but WILL NOT be used because there is no training.
        optimizer = optim.AdamW(
            multimodal_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )

        dummy_save_path = os.path.join(
            "/tmp",
            f"unused_fold{fold}.pth"
        )

        trainer = MultimodalTrainer(
            multimodal_model,
            args,
            criterion,
            optimizer,
            device,
            train_loader,
            test_loader,
            fold,
            dummy_save_path
        )

        # --------------------------------------------------
        # TEST ONLY
        # --------------------------------------------------

        print("\nTesting official checkpoint WITHOUT training...\n")

        f1_test, accuracy_test, precision_test, recall_test = \
            trainer.test_one_epoch(multimodal_model)

        test_metrics["F1 Score"].append(f1_test)
        test_metrics["Accuracy"].append(accuracy_test)
        test_metrics["Precision"].append(precision_test)
        test_metrics["Recall"].append(recall_test)

        print(f"\nFold {fold + 1} results:")
        print(f"F1 Score: {f1_test:.4f}")
        print(f"Accuracy: {accuracy_test:.4f}")
        print(f"Precision: {precision_test:.4f}")
        print(f"Recall: {recall_test:.4f}")

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n====================================")
    print("OFFICIAL PRETRAINED MODEL RESULTS")
    print("====================================")

    for metric, values in test_metrics.items():

        mean = np.mean(values)
        std = np.std(values)

        print(
            f"{metric}: Mean = {mean:.4f}, "
            f"Std = {std:.4f}"
        )


if __name__ == "__main__":
    main(args)
