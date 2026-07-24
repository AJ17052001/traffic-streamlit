import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Model Path
# -----------------------------------------------------------------------------
MODEL_PATH = "best.pt"

# -----------------------------------------------------------------------------
# Load YOLO Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please upload best.pt to your GitHub repository."
        )

    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load YOLO model.\n\n{e}")
    st.stop()

# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------
st.title("🚦 German Traffic Sign Recognition")
st.write("Upload a traffic sign image to detect traffic signs using YOLOv8.")

# -----------------------------------------------------------------------------
# Upload Image
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting traffic signs..."):

        results = model.predict(
            source=np.array(image),
            conf=0.25,
            verbose=False
        )

    result = results[0]
    plotted = result.plot()

    with col2:
        st.subheader("Detection Result")
        st.image(plotted, channels="BGR", use_container_width=True)

    st.markdown("---")
    st.subheader("Detection Details")

    if len(result.boxes) > 0:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            st.write(
                f"**{model.names[class_id]}** — {confidence:.2%}"
            )

    else:
        st.info("No traffic sign detected.")
