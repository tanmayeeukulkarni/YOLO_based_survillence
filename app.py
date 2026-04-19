import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T

st.title("Smart Surveillance Object Detection")

# Load model (stable, no external download issues)
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)
    model.eval()
    return model

model = load_model()

# Transform image
transform = T.Compose([T.ToTensor()])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image")

    img = transform(image)
    outputs = model([img])[0]

    # Draw boxes manually
    import numpy as np
    import cv2

    img_np = np.array(image)

    for box, score in zip(outputs['boxes'], outputs['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)

    st.image(img_np, caption="Detected Output")
