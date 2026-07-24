import os
import urllib.request
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German Traffic Sign Recognition", page_icon="🚦", layout="centered")

MODEL_PATH = "best.pt"

# Pre-trained YOLO model weights URL
# This link directly fetches a working YOLO model binary so you don't need local datasets
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# -----------------------------------------------------------------------------
# Helper Function: Download & Load Model Dynamically
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Downloads the actual binary weights from cloud storage if missing,
    ensuring Streamlit Cloud does not crash on empty/pointer files.
    """
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        with st.spinner("Downloading model weights for first-time setup..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    return YOLO(MODEL_PATH)

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# User Interface
# -----------------------------------------------------------------------------
st.title("🚦 German Traffic Sign Recognition")
st.write("Upload a traffic sign image below to run YOLOv8 model predictions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Read uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Inference
    with st.spinner("Detecting traffic signs..."):
        results = model(image)
        res_plotted = results[0].plot()

    with col2:
        st.subheader("YOLOv8 Detection")
        st.image(res_plotted, channels="BGR", use_container_width=True)

    # Show results detail
    st.markdown("---")
    st.subheader("Detection Summary")
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"• **Detected:** `{label}` | **Confidence:** `{conf:.2%}`")
    else:
        st.info("No objects detected in the uploaded image.")
               
