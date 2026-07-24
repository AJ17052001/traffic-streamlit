import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="Traffic Sign Detector",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 German Traffic Sign Recognition")
st.write("Upload a traffic sign image below to run YOLOv8 model predictions.")

# --- Load Model directly from local directory ---
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    # Verify the file exists locally
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Could not find '{MODEL_PATH}' in the root directory!")
        st.info("Make sure 'best.pt' is uploaded directly to your GitHub repository in the same folder as app.py.")
        st.stop()
        
    # Check if the file is just a git-lfs pointer file (1KB instead of ~6-15MB)
    if os.path.getsize(MODEL_PATH) < 100000:
        st.error("Error: 'best.pt' appears to be corrupted or a Git LFS pointer text file (< 100 KB).")
        st.info("Please delete 'best.pt' from GitHub and re-upload the real binary model file directly using 'Upload files' on GitHub.com.")
        st.stop()

    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# GTSRB Class Mapping Dictionary
gtsrb_classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing (vehicles over 3.5 tons)'
}

# --- Image Upload UI ---
uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png", "ppm"])

if uploaded_file is not None:
    # Load Image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Detect Signs 🚀"):
        with st.spinner("Analyzing image..."):
            # Convert to numpy/BGR format for OpenCV & YOLO
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Perform prediction using YOLO
            results = model.predict(source=img_bgr, conf=0.3, save=False)
            
            detected_items = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = gtsrb_classes.get(cls_id, f"Class {cls_id}")
                    detected_items.append(f"**{class_name}** — Confidence: `{conf * 100:.2f}%`")

            # Draw boxes on image
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Detected Output")
                st.image(res_rgb, use_container_width=True)

            st.write("---")
            if detected_items:
                st.markdown("### Detection Details:")
                for item in detected_items:
                    st.markdown(f"- {item}")
            else:
                st.warning("No traffic sign detected at confidence threshold 0.3.")
