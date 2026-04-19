import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("YOLO Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

file = st.file_uploader("Upload Image")

if file:
    img = Image.open(file)
    st.image(img)

    results = model(img)
    st.image(results[0].plot())
