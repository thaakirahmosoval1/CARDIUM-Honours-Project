"""CARDIUM image-only inference extension.

Place beside img_script/inference_img.py and invoke with the SAME CLI arguments
as the original. Runs the original inference implementation, intercepting its
patient-level metric inputs to save predictions, ROC plots, and metrics.

Does not change checkpoints, transforms, splits, or the original script.
Requires original inference_img.py to call compute_patient_metrics(y_true=..., y_score=...).
"""
import csv
import importlib.util
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / 'img_script' / 'inference_img.py'
OUT = ROOT / 'image_roc_results'


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not ORIGINAL.exists():
        raise FileNotFoundError(f'Original image inference script not found: {ORIGINAL}')
    spec = importlib.util.spec_from_file_location('cardium_original_image_inference', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'compute_patient_metrics'):
        raise RuntimeError('Original script does not expose compute_patient_metrics; inspect before running')
    if not hasattr(module, 'main') or not hasattr(module, 'get_main_parser'):
        raise RuntimeError('Original script must expose main(args) and get_main_parser()')

    original_metrics = module.compute_patient_metrics
    OUT.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    curves = []
    count = 0

    def capture_metrics(*args, **kwargs):
        nonlocal count
        # CARDIUM convention: y_score and y_true are passed as named arguments.
        if 'y_true' not in kwargs or 'y_score' not in kwargs:
            raise RuntimeError('Expected named y_true and y_score in compute_patient_metrics call')
        y_true = np.asarray(kwargs['y_true']).reshape(-1)
        y_score = np.asarray(kwargs['y_score']).reshape(-1)
        if len(y_true) == 0 or len(y_true) != len(y_score):
            raise ValueError('Empty or mismatched prediction arrays')
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError('Expected binary CHD labels 0/1')
        if not np.all(np.isfinite(y_score)) or not np.all((y_score >= 0) & (y_score <= 1)):
            raise ValueError('Expected finite CHD probabilities in [0,1]')

        count += 1
        fold = int(kwargs.get('fold', count - 1)) + 1
        if any(row['fold'] == fold for row in fold_rows):
            raise RuntimeError(f'Fold {fold} metrics called multiple times; check original script')
        predicted = (y_score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true.astype(int), predicted, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else float('nan')
        auc = float('nan')
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            curves.append((fold, fpr, tpr, auc))
            fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
            ax.plot(fpr, tpr, label=f'Fold {fold} (AUC = {auc:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
            ax.set(xlabel='False Positive Rate (1 − Specificity)', ylabel='True Positive Rate (Sensitivity)',
                   title=f'Image-only CARDIUM — Test ROC, Fold {fold}', xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(loc='lower right')
            fig.tight_layout()
            fig.savefig(OUT / f'fold_{fold}_roc.png')
            plt.close(fig)
        else:
            print(f'WARNING: Fold {fold} has one class; ROC-AUC is undefined')

        predictions = [dict(patient_row_index=i, y_true=int(label), y_score=float(score),
                            y_pred_at_0_5=int(pred))
                       for i, (label, score, pred) in enumerate(zip(y_true, y_score, predicted))]
        write_csv(OUT / f'fold_{fold}_predictions.csv', predictions,
                  ['patient_row_index', 'y_true', 'y_score', 'y_pred_at_0_5'])

        result = original_metrics(*args, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) not in (4, 5):
            raise ValueError('Unexpected return format from compute_patient_metrics')
        f1, accuracy, precision, recall = result[:4]
        fold_rows.append(dict(fold=fold, n_patients=len(y_true), tn=int(tn), fp=int(fp),
                              fn=int(fn), tp=int(tp), precision=float(precision),
                              recall=float(recall), **{'f1-score': float(f1)},
                              accuracy=float(accuracy), specificity=float(specificity), roc_auc=auc))
        print(f'Fold {fold}: specificity={specificity:.4f}, ROC-AUC={auc:.4f}, '
              f'TN={tn}, FP={fp}, FN={fn}, TP={tp}')
        # Preserve the return signature expected by the original image script.
        return result

    module.compute_patient_metrics = capture_metrics
    try:
        args = module.get_main_parser()
        module.main(args)
    finally:
        module.compute_patient_metrics = original_metrics
        if fold_rows:
            fold_rows.sort(key=lambda row: row['fold'])
            fields = ['fold', 'n_patients', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall',
                      'f1-score', 'accuracy', 'specificity', 'roc_auc']
            write_csv(OUT / 'image_metrics.csv', fold_rows, fields)
            summary = []
            for metric in fields[6:]:
                values = np.array([r[metric] for r in fold_rows], dtype=float)
                values = values[np.isfinite(values)]
                summary.append(dict(metric=metric,
                                    mean=float(np.mean(values)) if len(values) else float('nan'),
                                    std=float(np.std(values)) if len(values) else float('nan')))
            write_csv(OUT / 'image_metrics_summary.csv', summary, ['metric', 'mean', 'std'])
            if curves:
                fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
                for fold, fpr, tpr, auc in sorted(curves):
                    ax.plot(fpr, tpr, linewidth=2, label=f'Fold {fold} (AUC = {auc:.3f})')
                ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
                ax.set(xlabel='False Positive Rate (1 − Specificity)',
                       ylabel='True Positive Rate (Sensitivity)',
                       title='Image-only CARDIUM — Patient-Level Test ROC',
                       xlim=(0, 1), ylim=(0, 1.02))
                ax.legend(loc='lower right')
                fig.tight_layout()
                fig.savefig(OUT / 'image_roc_comparison.png')
                plt.close(fig)
            print(f'Image inference results saved in {OUT}')


if __name__ == '__main__':
    main()
