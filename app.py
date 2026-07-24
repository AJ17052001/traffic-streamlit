import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Traffic Sign Detector (YOLOv8)", layout="centered")

# ---------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

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


def predict_sign(input_image: Image.Image, conf_threshold: float = 0.5):
    img = np.array(input_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model.predict(source=img_bgr, conf=conf_threshold, save=False)

    scale = 4
    width = int(img_bgr.shape[1] * scale)
    height = int(img_bgr.shape[0] * scale)
    img_resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)

    detected_labels = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{gtsrb_classes.get(cls, f'Class {cls}')} {conf:.2f}"
            detected_labels.append(label)
            x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return output_img, detected_labels


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🚦 Traffic Sign Detection (YOLOv8)")
st.write("Upload an image containing a traffic sign to detect and classify it.")

conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "ppm"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running detection..."):
        output_img, detected_labels = predict_sign(input_image, conf_threshold)

    st.image(output_img, caption="Detection result", use_container_width=True)

    if detected_labels:
        st.subheader("Detected signs")
        for label in detected_labels:
            st.write(f"- {label}")
    else:
        st.info("No sign detected above the confidence threshold.")
else:
    st.info("👆 Upload an image to get started.")
