import streamlit as st
import torch
from PIL import Image
import numpy as np

st.title("Smart Surveillance Object Detection (YOLO-based)")

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image")

    results = model(image)

    st.image(results.render()[0], caption="Detected Output")
