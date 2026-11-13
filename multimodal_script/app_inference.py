import json
import numpy as np
import torch
import torch.nn as nn
import pathlib
import sys 
import os 

CARDIUM_path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(CARDIUM_path))

from PIL import Image
from glob import glob

from data.transformations import transform_test
from img_script.img_models.get_img_model import ImageModel
from tabular_script.tab_models.get_tab_model import TabularModel
from multimodal_script.multimodal_models.get_multimodal_model import MultimodalModel
from utils import get_main_parser

PROCESSED_JSON = str(
    CARDIUM_path / "data" / "tabular_data" /
    "cardium_clinical_data_woe_wnm_standarized_f_normalized_anon.json"
)
# Override with environment variables to run outside RunPod
CHECKPOINT_DIR = os.environ.get(
    "CARDIUM_CHECKPOINT_DIR",
    "/workspace/Cardium/multimodal_script/"
    "multimodal_checkpoints/2026-08-24-08-18-24"
)
DATASET_ROOT = os.environ.get(
    "CARDIUM_DATASET_ROOT",
    "/workspace/Cardium/dataset/cardium_images"
)
TEST_IMAGE_PATH = (
    "/workspace/Cardium/dataset/cardium_images/"
    "fold_1/train/Non_CHD/aedxf00593rrfgkl0/"
    "26_aedxf00593rrfgkl0.png"
)


def load_patient_records(json_path=PROCESSED_JSON):
    """
    Load the already-preprocessed CARDIUM clinical records.
    """
    with open(json_path, "r") as f:
        patients = json.load(f)

    return patients


def build_patient_vector(patient):
    """
    Reproduce the exact tabular feature construction used by CardiumDataset.

    Expected output:
        numpy array with 97 float32 features
    """

    embedding = []

    # WoE-encoded categorical histories
    embedding.extend(patient.get("pathological_history_woe", []))
    embedding.extend(patient.get("hereditary_history_woe", []))
    embedding.extend(patient.get("pharmacological_history_woe", []))

    # Numerical + ordinal/binary clinical variables
    features = [
        "platelets",
        "hemoglobin",
        "hematocrit",
        "white_blood_cells",
        "neutrophils",
        "age",
        "gestational_week_of_imaging",
        "body_mass_index",
        "gestational_age",
        "chromosomal_abnormality",
        "screening_procedures",
        "thromboembolic_risk",
        "psychosocial_risk",
        "depression_screening",
        "tobacco_use",
        "nutritional_deficiencies",
        "alcohol_use",
        "physical_activity",
        "pregnancies",
        "vaginal_births",
        "cesarean_sections",
        "miscarriages",
        "ectopic_pregnancies",
    ]

    for feature in features:
        value = patient.get(feature, -1)

        if isinstance(value, list):
            embedding.extend(value)

        elif isinstance(value, (int, float)):
            embedding.append(value)

    vector = np.array(embedding, dtype=np.float32)

    if len(vector) != 97:
        raise ValueError(
            f"Expected 97 tabular features, but got {len(vector)} "
            f"for patient {patient.get('id')}"
        )

    return vector


def get_patient_by_id(patient_id, json_path=PROCESSED_JSON):
    """
    Find one patient by anonymised patient ID.
    """
    patients = load_patient_records(json_path)

    for patient in patients:
        if patient["id"] == patient_id:
            return patient

    raise ValueError(f"Patient '{patient_id}' not found.")


def patient_to_tensor(patient, device="cpu"):
    """
    Convert one patient's processed clinical record into
    a [1, 97] PyTorch tensor.
    """
    vector = build_patient_vector(patient)

    tensor = torch.tensor(
        vector,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor.to(device)

def image_to_tensor(image_path, device="cpu"):
    """
    Load and preprocess one ultrasound image using the same
    transform_test pipeline used during CARDIUM inference.
    """

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform_test(np.array(image))

    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(device)
def load_multimodal_model(args, checkpoint_dir, fold=0, device=None):
    """
    Build the multimodal CARDIUM model and load one fold checkpoint.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # -------------------------
    # Image encoder
    # -------------------------
    img_model = ImageModel(args).build_model().to(device)

    if args.img_checkpoint is not None:
        img_checkpoint_path = os.path.join(
            args.img_checkpoint,
            f"fold{fold}_best_model.pth"
        )

        print(f"Loading image checkpoint: {img_checkpoint_path}")

        checkpoint = torch.load(
            img_checkpoint_path,
            map_location=device,
            weights_only=True
        )

        img_model.load_state_dict(checkpoint, strict=False)

    if args.img_model == "medvit":
        img_model.proj_head = nn.Identity()
    elif args.img_model in ["resnet18", "resnet50"]:
        img_model.fc = nn.Identity()
    else:
        img_model.head = nn.Identity()

    # -------------------------
    # Tabular encoder
    # -------------------------
    tab_model = TabularModel(args).build_model().to(device)

    if args.tab_checkpoint is not None:
        tab_checkpoint_path = os.path.join(
            args.tab_checkpoint,
            f"fold{fold}_best_model.pth"
        )

        print(f"Loading tabular checkpoint: {tab_checkpoint_path}")

        checkpoint = torch.load(
            tab_checkpoint_path,
            map_location=device,
            weights_only=True
        )

        tab_model.load_state_dict(checkpoint, strict=False)

    tab_model.mlp = nn.Identity()

    # -------------------------
    # Multimodal model
    # -------------------------
    model = MultimodalModel(
        img_model=img_model,
        tab_model=tab_model,
        args=args
    ).build_model().to(device)

    multimodal_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"fold{fold}_best_model.pth"
    )

    print(f"Loading multimodal checkpoint: {multimodal_checkpoint_path}")

    checkpoint = torch.load(
        multimodal_checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("Multimodal model loaded successfully.")

    return model, device


def predict_patient_image(
    model,
    img_tensor,
    tab_tensor,
    device,
    threshold=0.5
):
    """
    Predict CHD probability for one image and one patient's
    clinical data.
    """

    model.eval()

    img_tensor = img_tensor.to(device)
    tab_tensor = tab_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor, tab_tensor).squeeze()
        probability = torch.sigmoid(logits).item()

    prediction = "CHD" if probability > threshold else "Non-CHD"

    return probability, prediction
def get_patient_image_paths(patient_id, fold=1, split="test", class_name=None):
    """
    Find all image files for one patient.

    Parameters
    ----------
    patient_id : str
        Anonymised patient ID.
    fold : int
        Fold number in the dataset folders (1, 2, or 3).
    split : str
        "train" or "test".
    class_name : str | None
        "CHD" or "Non_CHD". If None, both folders are searched.

    Returns
    -------
    list[str]
        Sorted list of image paths.
    """

    base = os.path.join(DATASET_ROOT, f"fold_{fold}", split)

    classes = [class_name] if class_name else ["CHD", "Non_CHD"]

    image_paths = []

    for cls in classes:
        patient_dir = os.path.join(base, cls, patient_id)

        if os.path.isdir(patient_dir):
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                image_paths.extend(glob(os.path.join(patient_dir, ext)))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(
            f"No images found for patient {patient_id} "
            f"in fold_{fold}/{split}"
        )

    return image_paths
def predict_patient(
    model,
    image_paths,
    tab_tensor,
    device,
    threshold=0.5
):
    """
    Run inference on all images belonging to one patient,
    then average the image-level probabilities.
    """

    image_probabilities = []

    model.eval()

    with torch.no_grad():

        for image_path in image_paths:

            img_tensor = image_to_tensor(
                image_path,
                device=device
            )

            tab_input = tab_tensor.to(device)

            logits = model(
                img_tensor,
                tab_input
            ).squeeze()

            probability = torch.sigmoid(logits).item()

            image_probabilities.append(probability)

            print(
                f"{os.path.basename(image_path)} "
                f"-> CHD probability: {probability:.4f}"
            )

    patient_probability = float(
        np.mean(image_probabilities)
    )

    prediction = (
        "CHD"
        if patient_probability > threshold
        else "Non-CHD"
    )

    return patient_probability, prediction, image_probabilities

if __name__ == "__main__":

    args = get_main_parser()

    # --------------------------------------------------
    # TEST PATIENT
    # --------------------------------------------------

    # Replace this with a patient ID from fold_1/test
    patient_id = "aedxf00003rrfgkl0"

    patient = get_patient_by_id(patient_id)

    true_label = (
        "CHD"
        if patient.get("CHD") == 1
        else "Non-CHD"
    )

    print(f"Testing patient: {patient_id}")
    print(f"True label: {true_label}")

    # --------------------------------------------------
    # TABULAR INPUT
    # --------------------------------------------------

    tab_tensor = patient_to_tensor(patient)

    print(f"Tabular tensor shape: {tab_tensor.shape}")

    # --------------------------------------------------
    # FIND ALL IMAGES FOR THIS PATIENT
    # --------------------------------------------------

    image_paths = get_patient_image_paths(
        patient_id=patient_id,
        fold=1,
        split="test"
    )

    print(f"Found {len(image_paths)} images for patient")

    # --------------------------------------------------
    # LOAD FOLD 0 MODEL
    # --------------------------------------------------

    model, device = load_multimodal_model(
        args=args,
        checkpoint_dir=CHECKPOINT_DIR,
        fold=0
    )

    # --------------------------------------------------
    # PATIENT-LEVEL INFERENCE
    # --------------------------------------------------

    patient_probability, prediction, image_probabilities = predict_patient(
        model=model,
        image_paths=image_paths,
        tab_tensor=tab_tensor,
        device=device,
        threshold=0.5
    )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n================================")
    print("CARDIUM PATIENT-LEVEL RESULT")
    print("================================")
    print(f"Patient ID: {patient_id}")
    print(f"Number of images: {len(image_paths)}")
    print(f"True label: {true_label}")
    print(f"Average CHD probability: {patient_probability:.4f}")
    print(f"Predicted class: {prediction}")
    print("================================")