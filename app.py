import os
import sys

# 1. Disable Ultralytics auto-install/requirement checks before imports
os.environ["AUTOINSTALL"] = "false"
os.environ["YOLO_AUTOINSTALL"] = "false"

# 2. Register base directory so Python resolves local 'nets' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import urllib.request
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# 3. Explicitly disable Ultralytics settings auto-install
SETTINGS.update({"auto_install": False})

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

MODEL_PATH = "best.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# -----------------------------------------------------------------------------
# Helper Function: Download & Load YOLO Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """
    Ensures model weights exist locally as valid binary files.
    Downloads pre-trained weights if missing or corrupted (< 100 KB).
    """
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        with st.spinner("Downloading model weights..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            
    return YOLO(MODEL_PATH)

# Load the model safely
try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# User Interface & Detection Pipeline
# -----------------------------------------------------------------------------
st.title("🚦 German Traffic Sign Recognition")
st.write("Upload a traffic sign image below to run YOLOv8 model predictions.")

uploaded_file = st.file_uploader(
    "Upload an image", 
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running YOLOv8 detection..."):
        results = model(image)
        res_plotted = results[0].plot()

    with col2:
        st.subheader("YOLOv8 Detection")
        st.image(res_plotted, channels="BGR", use_container_width=True)

    # Output Detection Details
    st.markdown("---")
    st.subheader("Detection Details")
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"• **Detected Object:** `{label}` | **Confidence:** `{conf:.2%}`")
    else:
        st.info("No objects detected in the uploaded image.")
