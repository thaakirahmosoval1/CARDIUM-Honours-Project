import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os 
import numpy as np
import random
import json
from torch.utils.data import Subset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

def create_image_dataloaders(
    dataset_dir, dataset_class,
    transform_train, transform_test,
    batch_size, args
):
    fold_name = os.path.basename(os.path.normpath(dataset_dir))
    split_path = os.path.join(
        "experiment_splits",
        f"{fold_name}_split.json"
    )

    with open(split_path) as f:
        split = json.load(f)

    train_ids = set(split["train_patient_ids"])
    val_ids = set(split["val_patient_ids"])

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    train_dataset = dataset_class(
        root=train_dir, transform=transform_train
    )
    val_dataset = dataset_class(
        root=train_dir, transform=transform_test
    )
    test_dataset = dataset_class(
        root=test_dir, transform=transform_test
    )

    train_indices = [
        i for i, item in enumerate(train_dataset.data)
        if item[1] in train_ids
    ]
    val_indices = [
        i for i, item in enumerate(val_dataset.data)
        if item[1] in val_ids
    ]

    assert len(train_indices) > 0
    assert len(val_indices) > 0
    assert set(train_indices).isdisjoint(val_indices)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    if args.sampling:
        labels = [
            train_dataset.labels[i] for i in train_indices
        ]
        counts = np.bincount(labels, minlength=2)
        assert np.all(counts > 0)

        class_weights = 1.0 / torch.tensor(
            counts, dtype=torch.float32
        )
        sample_weights = class_weights[
            torch.tensor(labels, dtype=torch.long)
        ]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            sampler=sampler, num_workers=8,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            shuffle=True, num_workers=8,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size,
        shuffle=False, num_workers=8,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=8,
        pin_memory=True
    )

    print(
        f"{fold_name}: "
        f"{len(train_subset)} train images, "
        f"{len(val_subset)} validation images, "
        f"{len(test_dataset)} test images"
    )

    return train_loader, val_loader, test_loader