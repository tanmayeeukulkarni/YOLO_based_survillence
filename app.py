import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Smart Surveillance Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image")

    img = np.array(image)
    results = model(img)

    output = results[0].plot()

    st.image(output, caption="Detected Output")
