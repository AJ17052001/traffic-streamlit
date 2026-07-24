import os
import sys
import urllib.request
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# Path Fix for 'nets' Module (Prevents ModuleNotFoundError)
# -----------------------------------------------------------------------------
try:
    import tf_slim
    sys.path.append(os.path.join(os.path.dirname(tf_slim.__file__), "nets"))
except ImportError:
    pass

from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Streamlit App Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German Traffic Sign Recognition", 
    page_icon="🚦", 
    layout="centered"
)

MODEL_PATH = "best.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# -----------------------------------------------------------------------------
# Dynamic Model Downloader & Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """
    Downloads binary weights if missing or corrupted pointer text file.
    """
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        with st.spinner("Downloading pre-trained model weights for first-time launch..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            
    return YOLO(MODEL_PATH)

# Load the YOLO model
try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# User Interface
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
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection..."):
        results = model(image)
        res_plotted = results[0].plot()

    with col2:
        st.subheader("YOLOv8 Detection")
        st.image(res_plotted, channels="BGR", use_container_width=True)

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
        st.info("No traffic signs detected in the image.")
