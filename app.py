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
st.write("Upload a traffic sign image below to run model predictions.")

# --- Load Model (Cached to prevent reload on user interaction) ---
@st.cache_resource
def load_model():
    # Replace 'best.pt' with your model path if named differently
    model = YOLO("best.pt")
    return model

model = load_model()

# GTSRB Class Dictionary
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

# --- File Uploader ---
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "ppm"])

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Traffic Sign"):
        with st.spinner("Processing image..."):
            # Convert PIL image to BGR for OpenCV / YOLO processing
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Perform prediction
            results = model.predict(source=img_bgr, conf=0.3, save=False)
            
            detected_items = []
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = gtsrb_classes.get(cls_id, f"Class {cls_id}")
                    detected_items.append(f"**{class_name}** — Confidence: `{conf * 100:.2f}%`")
            
            # Draw bboxes / annotations
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.subheader("Detection Result")
            st.image(res_rgb, use_column_width=True)
            
            if detected_items:
                st.markdown("### Detected Classes:")
                for item in detected_items:
                    st.markdown(f"- {item}")
            else:
                st.warning("No traffic sign detected above confidence threshold.")
