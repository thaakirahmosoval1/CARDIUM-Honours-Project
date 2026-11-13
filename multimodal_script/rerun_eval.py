import sys, os, pathlib
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from collections import defaultdict

CARDIUM_path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(CARDIUM_path))

from multimodal_script.multimodal_models.get_multimodal_model import MultimodalModel
from data.load_multimodal_data import create_multimodal_dataloaders
from utils import *
from data.transformations import transform_train, transform_test
from img_script.img_models.get_img_model import ImageModel
from tabular_script.tab_models.get_tab_model import TabularModel
from data.dataloader import CardiumDataset

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = get_main_parser()
CKPT_DIR = os.environ["CKPT_DIR"]
OUT_DIR = os.environ.get("OUT_DIR", "rerun_results")
THRESHOLD = 0.5  # same as compute_patient_metrics (y_score > threshold)
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

rows, curves = [], []
for fold in range(args.folds):
    set_seed(42)
    _, test_loader = create_multimodal_dataloaders(
        dataset_dir=f"{args.image_folder_path}/fold_{fold+1}",
        json_root=args.json_path,
        dataset_class=CardiumDataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        args=args,
    )

    img_model = ImageModel(args).build_model().to(device)
    if args.img_model == "medvit":
        img_model.proj_head = nn.Identity()
    elif args.img_model in ["resnet18", "resnet50"]:
        img_model.fc = nn.Identity()
    else:
        img_model.head = nn.Identity()
    tab_model = TabularModel(args).build_model().to(device)
    tab_model.mlp = nn.Identity()

    model = MultimodalModel(img_model=img_model, tab_model=tab_model, args=args).build_model().to(device)
    ckpt = os.path.join(CKPT_DIR, f"fold{fold}_best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))  # strict: errors if config differs
    model.eval()

    y_true_p, y_score_p = {}, defaultdict(list)
    with torch.no_grad():
        for inputs, targets in test_loader:
            img, tab, pids = inputs[0].to(device), inputs[1].to(device), inputs[2]
            probs = torch.sigmoid(model(img, tab).squeeze(1)).cpu().numpy()
            for pid, p, t in zip(pids, probs, targets.numpy()):
                y_score_p[pid].append(p)
                y_true_p[pid] = t

    ids = list(y_true_p.keys())
    y_true = np.array([y_true_p[k] for k in ids])
    y_score = np.array([np.mean(y_score_p[k]) for k in ids])
    y_pred = (y_score > THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    rows.append(dict(fold=fold + 1, n_patients=len(y_true), tn=tn, fp=fp, fn=fn, tp=tp,
                     precision=precision, recall=recall, **{"f1-score": f1},
                     accuracy=(tn + tp) / len(y_true), specificity=tn / (tn + fp),
                     roc_auc=roc_auc_score(y_true, y_score)))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    curves.append((fold + 1, fpr, tpr, rows[-1]["roc_auc"]))
    np.savez(os.path.join(OUT_DIR, f"preds_fold{fold+1}.npz"), y_true=y_true, y_score=y_score, ids=np.array(ids))
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "multimodal_metrics.csv"), index=False)
metrics = ["precision", "recall", "f1-score", "accuracy", "specificity", "roc_auc"]
summ = pd.DataFrame({"metric": metrics,
                     "mean": [df[m].mean() for m in metrics],
                     "std": [df[m].std(ddof=0) for m in metrics]})
summ.to_csv(os.path.join(OUT_DIR, "multimodal_metrics_summary.csv"), index=False)
print(summ.to_string(index=False))

plt.figure(figsize=(7.5, 6.5))
for f, fpr, tpr, a in curves:
    plt.plot(fpr, tpr, lw=2, label=f"Fold {f} (AUC = {a:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
plt.xlim(0, 1); plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Multimodal CARDIUM — Patient-Level Test ROC")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUT_DIR, "multimodal_roc_comparison.png"), dpi=150, bbox_inches="tight")
