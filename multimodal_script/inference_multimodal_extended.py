"""Extended CARDIUM multimodal test inference with patient-level ROC-AUC.

Keep this file inside multimodal_script/ alongside inference_multimodal.py.
Uses the repository's existing inference_multimodal() and
compute_patient_metrics() helpers without changing training or preprocessing.
"""
import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

CARDIUM_path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(CARDIUM_path))

from data.transformations import transform_train, transform_test
from data.dataloader import CardiumDataset
from img_script.img_models.get_img_model import ImageModel
from tabular_script.tab_models.get_tab_model import TabularModel
from multimodal_script.multimodal_models.get_multimodal_model import MultimodalModel
from data.load_multimodal_data import create_multimodal_dataloaders
from utils import *


def load_checkpoint(model, checkpoint_path, device, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=strict)
    print(f'Model weights loaded from {checkpoint_path}')
    return model


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    output_dir = CARDIUM_path / 'multimodal_roc_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Inference device: {device}')

    metrics_folds = {key: [] for key in
                     ('precision', 'recall', 'f1-score', 'accuracy', 'specificity', 'roc_auc')}
    fold_rows = []
    combined_curves = []

    for fold in range(args.folds):
        print(f'\nProcessing fold {fold + 1}/{args.folds}')
        set_seed(42)
        if args.trimester is None:
            dataset_path = f'{args.image_folder_path}/fold_{fold+1}'
        else:
            dataset_path = f'{args.image_folder_path}/{args.trimester}_trimester/fold_{fold+1}'

        _, test_loader = create_multimodal_dataloaders(
            dataset_dir=dataset_path,
            json_root=args.json_path,
            dataset_class=CardiumDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            args=args,
        )

        img_model = ImageModel(args).build_model().to(device)
        if args.img_checkpoint:
            path = os.path.join(args.img_checkpoint, f'fold{fold}_best_model.pth')
            img_model = load_checkpoint(img_model, path, device, strict=False)

        if args.img_model == 'medvit':
            img_model.proj_head = torch.nn.Identity()
        elif args.img_model in ('resnet18', 'resnet50'):
            img_model.fc = torch.nn.Identity()
        else:
            img_model.head = torch.nn.Identity()

        tab_model = TabularModel(args).build_model().to(device)
        if args.tab_checkpoint:
            path = os.path.join(args.tab_checkpoint, f'fold{fold}_best_model.pth')
            tab_model = load_checkpoint(tab_model, path, device, strict=False)
        tab_model.mlp = torch.nn.Identity()

        multimodal_model = MultimodalModel(
            img_model=img_model, tab_model=tab_model, args=args
        ).build_model().to(device)
        path = os.path.join(args.multimodal_checkpoint, f'fold{fold}_best_model.pth')
        multimodal_model = load_checkpoint(multimodal_model, path, device, strict=True)
        multimodal_model.eval()

        with torch.inference_mode():
            y_true, y_score = inference_multimodal(multimodal_model, test_loader, device)

        y_true = np.asarray(y_true).reshape(-1)
        y_score = np.asarray(y_score).reshape(-1)
        if len(y_true) != len(y_score) or len(y_true) == 0:
            raise ValueError(f'Fold {fold+1}: mismatched or empty labels/scores')
        if not np.all(np.isfinite(y_score)):
            raise ValueError(f'Fold {fold+1}: non-finite prediction scores')
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError(f'Fold {fold+1}: expected binary 0/1 patient labels')
        if not np.all((y_score >= 0) & (y_score <= 1)):
            raise ValueError(f'Fold {fold+1}: expected CHD probabilities between 0 and 1')

        # Existing CARDIUM metrics remain the source for precision/recall/F1/accuracy.
        existing = compute_patient_metrics(
            y_score=y_score, y_true=y_true,
            mode='test', fold=fold, log_wandb=False,
        )
        if len(existing) not in (4, 5):
            raise ValueError('Unexpected compute_patient_metrics return format')
        f1, accuracy, precision, recall = existing[:4]

        # Match the standard CARDIUM 0.50 threshold; verify if your utils uses another.
        threshold = 0.50
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else float('nan')
        auc_value = float('nan')
        if len(np.unique(y_true)) == 2:
            auc_value = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            combined_curves.append((fold+1, fpr, tpr, auc_value))
            fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
            ax.plot(fpr, tpr, linewidth=2, label=f'Fold {fold+1} (AUC = {auc_value:.3f})')
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance (AUC = 0.500)')
            ax.set(xlabel='False Positive Rate (1 - Specificity)',
                   ylabel='True Positive Rate (Sensitivity)',
                   title=f'Multimodal CARDIUM — Test ROC, Fold {fold+1}',
                   xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(loc='lower right')
            fig.tight_layout()
            fig.savefig(output_dir / f'fold_{fold+1}_roc.png')
            plt.close(fig)
        else:
            print(f'WARNING: Fold {fold+1} has only one class; ROC-AUC undefined')

        predictions = [
            {'patient_row_index': i, 'y_true': int(label),
             'y_score': float(score), 'y_pred_at_0_5': int(pred)}
            for i, (label, score, pred) in enumerate(zip(y_true, y_score, y_pred))
        ]
        write_csv(output_dir / f'fold_{fold+1}_predictions.csv',
                  ['patient_row_index', 'y_true', 'y_score', 'y_pred_at_0_5'], predictions)

        metrics = dict(precision=float(precision), recall=float(recall),
                       **{'f1-score': float(f1)}, accuracy=float(accuracy),
                       specificity=float(specificity), roc_auc=auc_value)
        for key, value in metrics.items():
            metrics_folds[key].append(value)
        fold_rows.append({'fold': fold+1, 'n_patients': len(y_true),
                          'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                          **metrics})
        print(f'Fold {fold+1}: specificity={specificity:.4f}, ROC-AUC={auc_value:.4f}, '
              f'TN={tn}, FP={fp}, FN={fn}, TP={tp}')

    if combined_curves:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
        for fold_num, fpr, tpr, auc_value in combined_curves:
            ax.plot(fpr, tpr, linewidth=2, label=f'Fold {fold_num} (AUC = {auc_value:.3f})')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
        ax.set(xlabel='False Positive Rate (1 - Specificity)',
               ylabel='True Positive Rate (Sensitivity)',
               title='Multimodal CARDIUM — Patient-Level Test ROC',
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(output_dir / 'multimodal_roc_comparison.png')
        plt.close(fig)

    metric_names = list(metrics_folds)
    write_csv(output_dir / 'multimodal_metrics.csv',
              ['fold', 'n_patients', 'tn', 'fp', 'fn', 'tp'] + metric_names, fold_rows)
    summary_rows = []
    print('\nFinal average test metrics')
    for metric, values in metrics_folds.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        mean = float(np.mean(finite)) if len(finite) else float('nan')
        std = float(np.std(finite)) if len(finite) else float('nan')
        summary_rows.append({'metric': metric, 'mean': mean, 'std': std})
        print(f'{metric}: Mean = {mean:.4f}, Std = {std:.4f}')
    write_csv(output_dir / 'multimodal_metrics_summary.csv',
              ['metric', 'mean', 'std'], summary_rows)
    print(f'\nSaved results to: {output_dir}')


if __name__ == '__main__':
    args = get_main_parser()
    main(args)
