import streamlit as st

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

import torch
from torchvision import transforms
from PIL import Image
from Model_Architechture import BrainTumorModel

@st.cache_resource
def load_model():
    model = BrainTumorModel(num_classes=17)
    model.load_state_dict(torch.load("brain_tumor_trained_f.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

label_map = {
    0: "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1",
    1: "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+",
    2: "Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2",
    3: "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1",
    4: "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+",
    5: "Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2",
    6: "NORMAL T1",
    7: "NORMAL T2",
    8: "Neurocitoma (Central - Intraventricular, Extraventricular) T1",
    9: "Neurocitoma (Central - Intraventricular, Extraventricular) T1C+",
    10: "Neurocitoma (Central - Intraventricular, Extraventricular) T2",
    11: "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1",
    12: "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+",
    13: "Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2",
    14: "Schwannoma (Acustico, Vestibular - Trigeminal) T1",
    15: "Schwannoma (Acustico, Vestibular - Trigeminal) T1C+",
    16: "Schwannoma (Acustico, Vestibular - Trigeminal) T2"
}

transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Brain Tumor Detection Model 🧠</h1>", unsafe_allow_html=True)
st.write(
    "Upload an **MRI scan** below to classify the brain tumor type. "
    "Our model will predict the tumor category in just a few seconds!"
)

st.sidebar.title("About")
st.sidebar.info(
    """
    **Brain Tumor Classification App**
    - **Model:** Custom CNN
    - **Classes:** 17 tumor types
    - **Powered by:** PyTorch & Streamlit
    - **Created by:** Tarun Tiwari
    """
)

uploaded_file = st.file_uploader(
    "Upload MRI Image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False,
    help="Upload an MRI scan to analyze."
)

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return label_map.get(predicted.item(), "Unknown Tumor Type")

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(Image.open(uploaded_file).convert("RGB"), caption="Uploaded MRI", use_column_width=True)

    with col2:
        st.markdown("<h3 style='color: #FF5722;'>Classifying...</h3>", unsafe_allow_html=True)
        prediction = predict(Image.open(uploaded_file).convert("RGB"))
        st.success(f"Prediction: **{prediction}**")

#Evaluation Metrices
st.markdown("<h2 style='text-align: center; color: #2196F3;'>Evaluation Metrics</h2>", unsafe_allow_html=True)

metrics_container = st.container()

with metrics_container:
    with st.expander("F1 Scores"):
        f1_scores = {
            "Class 0": 0.9040,
            "Class 1": 0.9107,
            "Class 2": 0.8125,
            "Class 3": 0.8358,
            "Class 4": 0.9508,
            "Class 5": 0.7642,
            "Class 6": 0.8154,
            "Class 7": 0.8261,
            "Class 8": 0.8986,
            "Class 9": 0.9444,
            "Class 10": 0.8511,
            "Class 11": 0.9310,
            "Class 12": 1.0000,
            "Class 13": 0.9231,
            "Class 14": 0.7692,
            "Class 15": 0.8696,
            "Class 16": 0.7234
        }
        for class_name, score in f1_scores.items():
            st.markdown(f"<div style='background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 5px; text-align: center;'>"
                        f"<strong>{class_name}</strong>: {score:.2f}</div>", unsafe_allow_html=True)

with metrics_container:
    with st.expander("Specificity"):
        st.markdown("<h3 style='text-align: center;'>Specificity</h3>", unsafe_allow_html=True)

        specificity_scores = {
            "Class 0": 0.9938,
            "Class 1": 0.9761,
            "Class 2": 0.9927,
            "Class 3": 0.9891,
            "Class 4": 0.9961,
            "Class 5": 0.9880,
            "Class 6": 0.9739,
            "Class 7": 0.9737,
            "Class 8": 0.9954,
            "Class 9": 0.9953,
            "Class 10": 0.9954,
            "Class 11": 1.0000,
            "Class 12": 1.0000,
            "Class 13": 0.9977,
            "Class 14": 0.9988,
            "Class 15": 1.0000,
            "Class 16": 0.9943
        }

        selected_class = st.selectbox("Select a class to see its specificity:", list(specificity_scores.keys()))

        st.markdown(f"<div style='background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 5px; text-align: center;'>"
                    f"<strong>{selected_class}</strong>: {specificity_scores[selected_class]:.2f}</div>", unsafe_allow_html=True)


with st.expander("Confusion Matrix"):
    st.image("Screenshot 2024-10-13 235902.png", caption="Confusion Matrix", use_column_width=True)

with st.expander("Calibration Curve"):
    st.image("Screenshot 2024-10-13 235842.png", caption="Calibration Curve", use_column_width=True)

st.write("Model is still incomplete and need to be added more features like confimations, symptoms, etc.")
