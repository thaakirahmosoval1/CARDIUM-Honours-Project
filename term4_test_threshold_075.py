"""Read-only final test evaluation at threshold 0.75, selected using validation.
Run from /workspace/Cardium. No training or checkpoint changes.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from data.transformations import transform_train, transform_test
from data.load_img_data import create_image_dataloaders
from data.dataloader import CardiumDataset
from img_script.img_models.get_img_model import ImageModel
from utils import inference_image_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='img_script/image_checkpoints/2026-11-04-23-48-57')
    parser.add_argument('--image_folder_path', default='dataset/cardium_images')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    root = Path.cwd()
    ckpt_dir = root / args.checkpoint_dir
    assert ckpt_dir.is_dir(), f'Missing checkpoint directory: {ckpt_dir}'
    for fold in range(3):
        assert (ckpt_dir / f'fold{fold}_best_model.pth').is_file(), f'Missing fold {fold} checkpoint'

    # These parameters reproduce the saved ViT-Small architecture; no pretrained download.
    model_args = argparse.Namespace(img_model='vit_small', img_pretrain=False,
                                    n_classes=1, img_path_dropout=0.3,
                                    img_class_dropout=0.2)
    loader_args = argparse.Namespace(sampling=False)  # No training iteration is performed.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_folds = []
    threshold = 0.75  # Selected on validation; do not tune using test data.

    for fold in range(3):
        dataset_dir = root / args.image_folder_path / f'fold_{fold + 1}'
        assert dataset_dir.is_dir(), f'Missing dataset folder: {dataset_dir}'
        _, _, test_loader = create_image_dataloaders(
            dataset_dir=str(dataset_dir), dataset_class=CardiumDataset,
            transform_train=transform_train, transform_test=transform_test,
            batch_size=args.batch_size, args=loader_args)
        model = ImageModel(model_args).build_model().to(device)
        weights = torch.load(ckpt_dir / f'fold{fold}_best_model.pth',
                             map_location=device, weights_only=True)
        model.load_state_dict(weights, strict=True)
        y_true, y_score = inference_image_model(model, test_loader, device)
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        y_score = np.asarray(y_score).reshape(-1)
        assert len(y_true) == len(y_score) and len(y_true) > 0
        assert np.all(np.isfinite(y_score))
        print(f'Fold {fold+1}: {len(y_true)} test patients; {sum(y_true)} CHD')
        all_folds.append((y_true, y_score))
        del model

    print('\nTEST RESULTS — fixed validation-selected threshold = 0.75')
    print('Fold   Precision   Recall     F1       Accuracy')
    metrics = []
    for fold, (y_true, y_score) in enumerate(all_folds, start=1):
        pred = (y_score > threshold).astype(int)
        vals = [
            precision_score(y_true, pred, zero_division=0),
            recall_score(y_true, pred, zero_division=0),
            f1_score(y_true, pred, zero_division=0),
            accuracy_score(y_true, pred),
        ]
        metrics.append(vals)
        print(f'{fold:4d}   {vals[0]:9.4f}   {vals[1]:6.4f}   {vals[2]:6.4f}   {vals[3]:8.4f}')
    avg = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    print(f'Mean   {avg[0]:9.4f}   {avg[1]:6.4f}   {avg[2]:6.4f}   {avg[3]:8.4f}')
    print(f'Std    {std[0]:9.4f}   {std[1]:6.4f}   {std[2]:6.4f}   {std[3]:8.4f}')
    print('Threshold was selected on validation data, not test data.')


if __name__ == '__main__':
    main()
