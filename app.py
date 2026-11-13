import streamlit as st
import os

from multimodal_script.app_inference import (
    load_patient_records,
    patient_to_tensor,
    get_patient_image_paths,
    load_multimodal_model,
    predict_patient,
    CHECKPOINT_DIR,
    DATASET_ROOT
)

from utils import get_main_parser

# --------------------------------------------------
# Edit these before submitting
# --------------------------------------------------
STUDENT_NAME = "Thaakirah Mosoval"
STUDENT_NUMBER = "4314422"

ROSE = "#B83B5E"
ROSE_LIGHT = "#FCE8EC"

st.set_page_config(
    page_title="CARDIUM CHD Detection",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{ color: {ROSE}; }}
    [data-testid="stMetric"] {{
        background: {ROSE_LIGHT};
        border-radius: 10px;
        padding: 12px 16px;
    }}
    .cardium-disclaimer {{
        background: {ROSE_LIGHT};
        border-left: 6px solid {ROSE};
        border-radius: 6px;
        padding: 14px 18px;
        margin: 8px 0 20px 0;
        color: #2D2A32;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("❤️ CARDIUM: Prenatal CHD Detection")
st.write(
    "Multimodal inference using fetal ultrasound images "
    "and maternal clinical data."
)

st.markdown(
    """
    <div class="cardium-disclaimer">
    <strong>Not for diagnosis.</strong> This application is a research
    demonstration built for an Honours project. Its purpose is to show how
    fetal ultrasound images and clinical data can be combined to
    <em>support</em> a clinician's decision-making. It is not a medical
    device and has not been clinically validated. It can miss cases of
    congenital heart disease and can flag healthy patients, so its output
    must never replace the judgement of a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar: how far to trust the output
# --------------------------------------------------

with st.sidebar:
    st.header("About the model")
    st.markdown(
        """
        Results from 3-fold cross-validation, evaluated per patient
        (about 25 CHD cases per fold):

        - **Recall 74.5%**: roughly 1 in 4 CHD cases is missed, so
          "Not flagged" does not rule out CHD.
        - **Specificity 96.8%**, **precision 65%**: roughly 1 in 3
          flagged patients is a false alarm.
        - **ROC AUC 0.950** (image-only model: 0.800).
        - Decision threshold: 0.5.

        These figures come from internal cross-validation only. The model
        has not been tested on an external dataset.
        """
    )

# --------------------------------------------------
# Cached loaders
# --------------------------------------------------


@st.cache_resource
def load_model(checkpoint_fold):
    args = get_main_parser()

    model, device = load_multimodal_model(
        args=args,
        checkpoint_dir=CHECKPOINT_DIR,
        fold=checkpoint_fold
    )

    return model, device


@st.cache_data
def load_patients():
    return {p["id"]: p for p in load_patient_records()}


def risk_status(probability):
    """Softer wording than a hard verdict, with an uncertainty band."""
    if probability >= 0.60:
        return "Flagged for review"
    if probability >= 0.40:
        return "Borderline"
    return "Not flagged"


patients = load_patients()

# --------------------------------------------------
# Select test fold, class and patient
# --------------------------------------------------

selected_fold = st.selectbox(
    "Select test fold",
    [1, 2, 3]
)

checkpoint_fold = selected_fold - 1

class_name = st.selectbox(
    "Select patient class",
    ["Non_CHD", "CHD"]
)

st.caption(
    "Patients shown here come from the study's held-out test folds and "
    "are used for demonstration only."
)

patient_folder = os.path.join(
    DATASET_ROOT,
    f"fold_{selected_fold}",
    "test",
    class_name
)

if not os.path.isdir(patient_folder):
    st.error(
        f"Dataset folder not found: {patient_folder}. "
        "Set the CARDIUM_DATASET_ROOT environment variable to the "
        "folder that contains fold_1, fold_2 and fold_3."
    )
    st.stop()

patient_ids = sorted(
    [
        name
        for name in os.listdir(patient_folder)
        if os.path.isdir(os.path.join(patient_folder, name))
    ]
)

if not patient_ids:
    st.warning("No patients found in this folder.")
    st.stop()

patient_id = st.selectbox(
    "Select patient",
    patient_ids
)

# --------------------------------------------------
# Load patient information and images
# --------------------------------------------------

patient = patients.get(patient_id)

if patient is None:
    st.error(f"Patient '{patient_id}' not found in the clinical records.")
    st.stop()

true_label = (
    "CHD"
    if patient.get("CHD") == 1
    else "Non-CHD"
)

st.write(f"**Patient ID:** {patient_id}")

image_paths = get_patient_image_paths(
    patient_id=patient_id,
    fold=selected_fold,
    split="test",
    class_name=class_name
)
st.write(f"**Number of ultrasound images:** {len(image_paths)}")

with st.expander("View ultrasound images"):
    cols = st.columns(3)

    for i, image_path in enumerate(image_paths):
        with cols[i % 3]:
            st.image(
                image_path,
                caption=os.path.basename(image_path),
                use_container_width=True
            )

# --------------------------------------------------
# Run inference (results are kept in session state so they
# survive reruns caused by other widgets)
# --------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = {}

result_key = f"{selected_fold}|{class_name}|{patient_id}"

if st.button("Run CHD Prediction", type="primary"):

    with st.spinner("Running CARDIUM inference..."):

        model, device = load_model(checkpoint_fold)
        tab_tensor = patient_to_tensor(
            patient,
            device=device
        )

        patient_probability, prediction, image_probabilities = (
            predict_patient(
                model=model,
                image_paths=image_paths,
                tab_tensor=tab_tensor,
                device=device,
                threshold=0.5
            )
        )

    st.session_state.results[result_key] = {
        "probability": patient_probability,
        "prediction": prediction,
        "image_probabilities": image_probabilities,
    }

result = st.session_state.results.get(result_key)

if result:

    probability = result["probability"]

    st.subheader("Patient-Level Prediction")
    st.caption(
        "Decision-support output only. This is not a diagnosis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CHD Probability", f"{probability:.2%}")

    with col2:
        st.metric("Model output", risk_status(probability))

    with col3:
        st.metric("True Label", true_label)

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "Flagged at 60% or above, borderline between 40% and 60%. "
        "The 40-60% band is illustrative and has not been validated; "
        "the reported evaluation metrics use a 0.5 threshold."
    )

    if result["prediction"] == true_label:
        st.success("At the 0.5 threshold, the model output matches the true label.")
    else:
        st.warning(
            "At the 0.5 threshold, the model output differs from the true "
            "label. This is the kind of error the model can make."
        )

    # --------------------------------------------------
    # Ultrasound views ranked by score
    # --------------------------------------------------

    ranked = sorted(
        zip(image_paths, result["image_probabilities"]),
        key=lambda item: item[1],
        reverse=True
    )

    st.subheader("Highest-scoring ultrasound views")
    st.caption(
        "Each view is scored together with the same clinical data. "
        "The patient score is the mean across all views."
    )

    top_cols = st.columns(3)
    for col, (image_path, image_probability) in zip(top_cols, ranked[:3]):
        with col:
            st.image(
                image_path,
                caption=f"{os.path.basename(image_path)} | {image_probability:.1%}",
                use_container_width=True
            )

    with st.expander("All image predictions (highest first)"):
        for image_path, image_probability in ranked:
            st.write(
                f"{os.path.basename(image_path)}: "
                f"{image_probability:.4f}"
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    f"Developed by **{STUDENT_NAME}** | Student number: **{STUDENT_NUMBER}**  \n"
    "Computer Science Honours, University of the Western Cape | "
    "CARDIUM research project"
)
